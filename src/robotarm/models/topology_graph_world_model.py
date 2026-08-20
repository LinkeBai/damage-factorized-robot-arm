"""Shared chain-graph dynamics model for topology surgery.

Each joint is a node with shared parameters. Bidirectional messages propagate
the effect of a locked joint along the serial chain, while a pooled graph code
predicts the object state in Push.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .topology_surgery import TopologySurgery


@dataclass
class TopologyGraphConfig:
    dof: int = 5
    object_dim: int = 4
    hidden_dim: int = 96
    message_steps: int = 2


class TopologyGraphWorldModel(nn.Module):
    def __init__(self, cfg: TopologyGraphConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TopologyGraphConfig()
        c = self.cfg
        # q, qvel, action, locked, lock angle, normalized depth, object state
        node_input_dim = 6 + c.object_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
        )
        self.message = nn.Sequential(
            nn.Linear(c.hidden_dim * 2, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.update = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.temporal = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.joint_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, 2)
        )
        self.object_head = nn.Sequential(
            nn.Linear(c.hidden_dim + c.object_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.object_dim),
        )
        self.surgery = TopologySurgery()

    def _neighbor_sum(self, nodes: torch.Tensor) -> torch.Tensor:
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
        c = self.cfg
        state = self.surgery.project_state(state, mask, lock_angle)
        action = self.surgery.project_action(action, mask)
        q = state[:, : c.dof]
        qvel = state[:, c.dof : 2 * c.dof]
        obj = state[:, 2 * c.dof :]
        depth = torch.linspace(0.0, 1.0, c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, c.dof).expand(state.shape[0], -1)
        obj_nodes = obj.unsqueeze(1).expand(-1, c.dof, -1)
        features = torch.cat(
            (q.unsqueeze(-1), qvel.unsqueeze(-1), action.unsqueeze(-1),
             mask.unsqueeze(-1), lock_angle.unsqueeze(-1), depth.unsqueeze(-1), obj_nodes),
            dim=-1,
        )
        nodes = self.node_encoder(features)
        for _ in range(c.message_steps):
            messages = self.message(torch.cat((nodes, self._neighbor_sum(nodes)), dim=-1))
            nodes = self.update(
                messages.reshape(-1, c.hidden_dim), nodes.reshape(-1, c.hidden_dim)
            ).view_as(nodes)
        if hidden is None:
            hidden = torch.zeros_like(nodes)
        hidden = self.temporal(
            nodes.reshape(-1, c.hidden_dim), hidden.reshape(-1, c.hidden_dim)
        ).view_as(nodes)
        joint_delta = self.joint_head(hidden)
        next_q = q + joint_delta[..., 0]
        next_qvel = qvel + joint_delta[..., 1]
        pooled = hidden.mean(dim=1)
        next_obj = obj + self.object_head(torch.cat((pooled, obj), dim=-1))
        prediction = torch.cat((next_q, next_qvel, next_obj), dim=-1)
        prediction = self.surgery.project_state(prediction, mask, lock_angle)
        return prediction, hidden
