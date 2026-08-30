"""Low-capacity gated reaction head for the Gate H fidelity audit."""
from __future__ import annotations

import torch
from torch import nn

from .topology_graph_world_model import TopologyGraphWorldModel
from .topology_surgery import TopologySurgery


class GatedReactionGraph(nn.Module):
    """Correct a frozen graph model through a small, near-zero reaction path."""

    def __init__(
        self,
        base: TopologyGraphWorldModel,
        bottleneck_dim: int = 16,
        message_steps: int = 3,
        gate_logit_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        hidden_dim = base.cfg.hidden_dim
        self.message_steps = message_steps
        self.reaction_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, bottleneck_dim), nn.SiLU()
        )
        self.reaction_message = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, bottleneck_dim), nn.SiLU()
        )
        self.joint_correction = nn.Linear(bottleneck_dim, 2)
        self.object_correction = nn.Linear(bottleneck_dim, base.cfg.object_dim)
        self.joint_gate_logit = nn.Parameter(torch.tensor(gate_logit_init))
        self.object_gate_logit = nn.Parameter(torch.tensor(gate_logit_init))
        self.surgery = TopologySurgery()

    @staticmethod
    def _neighbor_sum(nodes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        result[:, 1:] += nodes[:, :-1]
        result[:, :-1] += nodes[:, 1:]
        return result

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
        lock_angle: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_mask = torch.zeros_like(mask)
        with torch.no_grad():
            base_prediction, base_hidden = self.base.step(
                state, action, zero_mask, zero_mask, hidden
            )
        dof = self.base.cfg.dof
        constraint_features = torch.stack(
            (
                (lock_angle - base_prediction[:, :dof]) * mask,
                -base_prediction[:, dof : 2 * dof] * mask,
                mask,
            ),
            dim=-1,
        )
        reaction = self.reaction_encoder(
            torch.cat((base_hidden, constraint_features), dim=-1)
        ) * mask.unsqueeze(-1)
        for _ in range(self.message_steps):
            reaction = reaction + self.reaction_message(
                torch.cat((reaction, self._neighbor_sum(reaction)), dim=-1)
            )

        joint_gate = torch.sigmoid(self.joint_gate_logit)
        object_gate = torch.sigmoid(self.object_gate_logit)
        correction = self.joint_correction(reaction) * joint_gate
        correction = correction * (1.0 - mask).unsqueeze(-1)
        prediction = base_prediction.clone()
        prediction[:, :dof] += correction[..., 0]
        prediction[:, dof : 2 * dof] += correction[..., 1]
        prediction[:, 2 * dof :] += (
            self.object_correction(reaction.mean(dim=1)) * object_gate
        )
        return self.surgery.project_state(prediction, mask, lock_angle), base_hidden
