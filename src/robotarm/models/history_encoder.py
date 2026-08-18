"""Amortized residual encoder (PROJECT-PLAN-V5 §4.3B, §6.4 ablation 8).

Variant B of residual-context inference: instead of optimizing a residual
latent via gradient descent at deployment (variant A, latent optimization), a
small GRU reads the recent ``(state, action)`` transition history and infers
the residual in a single forward pass.

Unlike variant A, this encoder conditions on the *same* topology encoder as
DFWM and only replaces the residual source: context = ``[e_topology, z]``,
where ``z`` comes from a forward pass rather than gradient optimization. This
makes it a fair amortized counterpart to latent optimization — the only
difference is inference cost vs. adaptation flexibility.

Design notes:
- Same pretraining data as DFWM (identical training trajectories).
- Same observation history (``(state_t, action_t)`` sequence).
- Same context structure (topology + 8-dim residual).
- Output dim is the residual dim (8), concatenated with topology by the caller.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["HistoryEncoder"]


class HistoryEncoder(nn.Module):
    """Encode a batch of transition trajectories into a residual context.

    Parameters
    ----------
    state_dim:
        Dimensionality of the proprioceptive state (default 10 for 5-DoF).
    action_dim:
        Dimensionality of the action vector (default 5).
    hidden_dim:
        GRU hidden size.
    out_dim:
        Residual dimension (default 8, concatenated with topology by caller).
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        hidden_dim: int = 64,
        out_dim: int = 8,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Map (K, T, state) + (K, T, action) histories to one (out_dim,) context.

        Each of the K trajectories is processed by the shared GRU; the per-
        trajectory final hidden states are mean-pooled, then mapped to the
        output context. The result is a single deployment-level context vector.
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        x = torch.cat([states, actions], dim=-1)  # (K, T, state+action)
        _, hidden = self.rnn(x)  # (1, K, hidden_dim)
        pooled = hidden.squeeze(0).mean(dim=0)  # (hidden_dim,)
        return self.head(pooled)  # (out_dim,)
