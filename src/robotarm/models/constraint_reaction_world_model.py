"""Constraint-reaction adapter over a frozen graph dynamics model."""
from __future__ import annotations

import torch
from torch import nn

from .topology_graph_world_model import TopologyGraphWorldModel
from .topology_surgery import TopologySurgery


class ConstraintReactionWorldModel(nn.Module):
    """Propagate predicted lock violations as learned reaction messages.

    The base graph model is frozen. The adapter sees only the constraint
    residual at locked nodes and local base hidden states, then corrects free
    joints and the object. This prevents an unconstrained latent from replacing
    the nominal dynamics.
    """

    def __init__(self, base: TopologyGraphWorldModel, message_steps: int = 3) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        hidden_dim = base.cfg.hidden_dim
        self.message_steps = message_steps
        self.reaction_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.reaction_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reaction_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.free_joint_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.object_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, base.cfg.object_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))
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
        zero_angle = torch.zeros_like(lock_angle)
        with torch.no_grad():
            base_prediction, base_hidden = self.base.step(
                state, action, zero_mask, zero_angle, hidden
            )
        dof = self.base.cfg.dof
        position_residual = (lock_angle - base_prediction[:, :dof]) * mask
        velocity_residual = -base_prediction[:, dof : 2 * dof] * mask
        constraint_features = torch.stack(
            (position_residual, velocity_residual, mask), dim=-1
        )
        reaction = self.reaction_encoder(
            torch.cat((base_hidden, constraint_features), dim=-1)
        )
        # Only locked nodes inject a reaction; messages carry it along the chain.
        reaction = reaction * mask.unsqueeze(-1)
        for _ in range(self.message_steps):
            message = self.reaction_message(
                torch.cat((reaction, self._neighbor_sum(reaction)), dim=-1)
            )
            reaction = self.reaction_update(
                message.reshape(-1, message.shape[-1]),
                reaction.reshape(-1, reaction.shape[-1]),
            ).view_as(reaction)
        correction = self.free_joint_correction(reaction) * self.scale
        free = (1.0 - mask).unsqueeze(-1)
        correction = correction * free
        prediction = base_prediction.clone()
        prediction[:, :dof] += correction[..., 0]
        prediction[:, dof : 2 * dof] += correction[..., 1]
        prediction[:, 2 * dof :] += self.object_correction(reaction.mean(dim=1)) * self.scale
        prediction = self.surgery.project_state(prediction, mask, lock_angle)
        return prediction, base_hidden
