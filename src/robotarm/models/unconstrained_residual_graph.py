"""Parameter-matched residual adapter without constraint-reaction structure."""
from __future__ import annotations

import torch
from torch import nn

from .topology_graph_world_model import TopologyGraphWorldModel


class UnconstrainedResidualGraph(nn.Module):
    """Same adapter capacity as constraint reaction, without lock semantics."""

    def __init__(self, base: TopologyGraphWorldModel, message_steps: int = 3) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        hidden_dim = base.cfg.hidden_dim
        self.message_steps = message_steps
        self.residual_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.residual_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.joint_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.object_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, base.cfg.object_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    @staticmethod
    def _neighbor_sum(nodes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        result[:, 1:] += nodes[:, :-1]
        result[:, :-1] += nodes[:, 1:]
        return result

    def step(
        self, state: torch.Tensor, action: torch.Tensor,
        mask: torch.Tensor, lock_angle: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mask, lock_angle
        batch = state.shape[0]
        zero_mask = torch.zeros(batch, 5, device=state.device, dtype=state.dtype)
        with torch.no_grad():
            base_prediction, base_hidden = self.base.step(
                state, action, zero_mask, zero_mask, hidden
            )
        q = state[:, :5]
        qvel = state[:, 5:10]
        features = torch.stack((q, qvel, action), dim=-1)
        residual = self.residual_encoder(torch.cat((base_hidden, features), dim=-1))
        for _ in range(self.message_steps):
            message = self.residual_message(
                torch.cat((residual, self._neighbor_sum(residual)), dim=-1)
            )
            residual = self.residual_update(
                message.reshape(-1, message.shape[-1]),
                residual.reshape(-1, residual.shape[-1]),
            ).view_as(residual)
        correction = self.joint_correction(residual) * self.scale
        prediction = base_prediction.clone()
        prediction[:, :5] += correction[..., 0]
        prediction[:, 5:10] += correction[..., 1]
        prediction[:, 10:] += self.object_correction(residual.mean(dim=1)) * self.scale
        return prediction, base_hidden
