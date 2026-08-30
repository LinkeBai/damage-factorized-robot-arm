"""Amortized residual encoder with A+B+C combined training signal.

A: Physics parameter supervision (RMA-style)
   encoder(traj) -> z -> linear head -> predict physics params
   Prevents posterior collapse, grounds z in real physics.
   Basis: RMA (Kumar et al. RSS 2021)

B: Contrastive learning (InfoNCE)
   Same physics profile -> similar z, different profile -> different z
   Forces encoder to be discriminative across profiles.
   Basis: NT-Xent / SimCLR (Chen et al. ICML 2020)

C: Active probing trajectories (at data collection time)
   Deterministic probing actions expose motor strength, damping, delay.
   More physics signal than random excitation.
   Basis: Active system identification literature.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualEncoder(nn.Module):
    """Encode K trajectories -> z_residual with auxiliary physics prediction head.

    Main output: z ∈ R^z_dim (for WM conditioning)
    Aux output:  physics_pred ∈ R^z_dim (for physics supervision during training)

    The aux head is only used during training. At test time, call forward() only.
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 5,
        hidden_dim: int = 128,
        z_dim: int = 8,
    ) -> None:
        super().__init__()
        transition_dim = state_dim + action_dim + state_dim

        # Shared transition encoder
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # z head: pooled hidden -> z
        self.z_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, z_dim),
        )
        # Aux A: physics parameter prediction head
        self.physics_head = nn.Linear(z_dim, z_dim)

        # Contrastive projection head (B): project z to unit sphere
        self.contrastive_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    def _encode_transitions(
        self,
        states: torch.Tensor,   # (K, T+1, state_dim)
        actions: torch.Tensor,  # (K, T,   action_dim)
    ) -> torch.Tensor:
        """Shared encoding: return pooled hidden (hidden_dim,)."""
        K, T, _ = actions.shape
        s_t = states[:, :-1, :]
        s_tp1 = states[:, 1:, :]
        delta_s = s_tp1 - s_t
        x = torch.cat([s_t, actions, delta_s], dim=-1)  # (K, T, transition_dim)
        x = x.view(K * T, -1)
        h = self.transition_encoder(x)                  # (K*T, hidden_dim)
        return h.mean(dim=0)                            # (hidden_dim,)

    def forward(
        self,
        states: torch.Tensor,   # (K, T+1, state_dim)
        actions: torch.Tensor,  # (K, T,   action_dim)
    ) -> torch.Tensor:
        """Return z_residual (z_dim,). Used at test time."""
        h = self._encode_transitions(states, actions)
        return self.z_head(h)

    def forward_train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return z + auxiliary outputs for training. Used at train time."""
        h = self._encode_transitions(states, actions)
        z = self.z_head(h)
        physics_pred = self.physics_head(z)
        z_proj = F.normalize(self.contrastive_proj(z), dim=-1)
        return {"z": z, "physics_pred": physics_pred, "z_proj": z_proj}


# ── Loss functions ─────────────────────────────────────────────────────────────

def physics_supervision_loss(
    physics_pred: torch.Tensor,  # (N, z_dim) predicted
    physics_target: torch.Tensor, # (N, z_dim) true normalized params
) -> torch.Tensor:
    """MSE between predicted and true normalized physics parameters (Loss A)."""
    return F.mse_loss(physics_pred, physics_target)


def infonce_loss(
    z_projs: list[torch.Tensor],   # list of (z_dim,) projected z vectors
    physics_labels: list[str],     # physics profile name per z
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss (Loss B).

    Positive pairs: same physics profile.
    Negative pairs: different physics profiles.
    Pulls same-physics z together, pushes different-physics z apart.
    """
    if len(z_projs) < 2:
        return torch.tensor(0.0, device=z_projs[0].device)

    z_mat = torch.stack(z_projs)  # (N, z_dim)
    # 数值稳定：归一化后再计算相似度
    z_mat = F.normalize(z_mat, dim=-1, eps=1e-8)
    N = z_mat.shape[0]

    # Similarity matrix
    sim = torch.mm(z_mat, z_mat.T) / temperature  # (N, N)

    # Positive mask: same physics profile, excluding self
    pos_mask = torch.zeros(N, N, dtype=torch.bool, device=z_mat.device)
    for i in range(N):
        for j in range(N):
            if i != j and physics_labels[i] == physics_labels[j]:
                pos_mask[i, j] = True

    if not pos_mask.any():
        return z_mat.sum() * 0.0  # 保持计算图，返回零

    # Remove self-similarity diagonal
    eye = torch.eye(N, dtype=torch.bool, device=z_mat.device)
    sim = sim.masked_fill(eye, float('-inf'))

    # InfoNCE: for each anchor, log(exp(pos_sim) / sum(exp(all_sim)))
    log_softmax = F.log_softmax(sim, dim=-1)  # (N, N)
    loss = -(log_softmax * pos_mask.float()).sum(dim=-1)
    n_pos = pos_mask.float().sum(dim=-1).clamp(min=1)
    loss = (loss / n_pos).mean()
    return loss
