"""Reduced-coordinate graph dynamics for known locked joints."""
from __future__ import annotations

import torch
from torch import nn

from .topology_graph_world_model import TopologyGraphConfig
from .topology_surgery import TopologySurgery


class ReducedCoordinateGraphWorldModel(nn.Module):
    """Run dynamics only on free joints and compact the active kinematic chain.

    Locked nodes carry no recurrent state and are omitted from object pooling.
    Their nearest free neighbors are connected across the removed coordinates.
    Full states are reconstructed analytically after every transition.
    """

    def __init__(
        self, cfg: TopologyGraphConfig | None = None, *,
        detach_object_features: bool = False, bridge_edge_features: bool = False,
        packed_active_nodes: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg or TopologyGraphConfig(hidden_dim=128)
        self.detach_object_features = detach_object_features
        self.bridge_edge_features = bridge_edge_features
        self.packed_active_nodes = packed_active_nodes
        c = self.cfg
        # q, qvel, action, normalized original-chain depth, object state
        node_input_dim = 4 + c.object_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
        )
        message_input_dim = 2 * c.hidden_dim + (5 if bridge_edge_features else 0)
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, c.hidden_dim), nn.SiLU(),
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

    @staticmethod
    def _compact_neighbor_sum(nodes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Connect consecutive active nodes after locked coordinates are removed."""
        result = torch.zeros_like(nodes)
        dof = nodes.shape[1]
        for left in range(dof):
            for right in range(left + 1, dof):
                weight = active[:, left] * active[:, right]
                if right > left + 1:
                    weight = weight * (1.0 - active[:, left + 1 : right]).prod(dim=1)
                weight = weight.unsqueeze(-1)
                result[:, left] += nodes[:, right] * weight
                result[:, right] += nodes[:, left] * weight
        return result

    @staticmethod
    def _compact_edge_features(
        active: torch.Tensor, lock_angle: torch.Tensor
    ) -> torch.Tensor:
        """Describe contracted edges, including the removed fixed rotation."""
        batch, dof = active.shape
        result = torch.zeros(batch, dof, 5, device=active.device, dtype=active.dtype)
        for left in range(dof):
            for right in range(left + 1, dof):
                weight = active[:, left] * active[:, right]
                if right > left + 1:
                    weight = weight * (1.0 - active[:, left + 1 : right]).prod(dim=1)
                span = float(right - left) / max(dof - 1, 1)
                bridge = float(right > left + 1)
                result[:, left, 0] += weight * span
                result[:, right, 0] += weight * span
                result[:, left, 1] += weight * bridge
                result[:, right, 1] += weight * bridge
                result[:, left, 2] += weight
                result[:, right, 2] -= weight
                if right > left + 1:
                    contracted_angle = (
                        lock_angle[:, left + 1 : right]
                        * (1.0 - active[:, left + 1 : right])
                    ).sum(dim=1)
                    result[:, left, 3] += weight * torch.sin(contracted_angle)
                    result[:, right, 3] += weight * torch.sin(contracted_angle)
                    result[:, left, 4] += weight * torch.cos(contracted_angle)
                    result[:, right, 4] += weight * torch.cos(contracted_angle)
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
        active = 1.0 - mask
        q = state[:, : c.dof]
        qvel = state[:, c.dof : 2 * c.dof]
        obj = state[:, 2 * c.dof :]
        depth = torch.linspace(0.0, 1.0, c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, c.dof).expand(state.shape[0], -1)
        original_indices = None
        if self.packed_active_nodes:
            # Stable sort moves active joints into contiguous reduced-coordinate slots.
            original_indices = torch.argsort(mask, dim=1, stable=True)
            q = q.gather(1, original_indices)
            qvel = qvel.gather(1, original_indices)
            action = action.gather(1, original_indices)
            depth = depth.gather(1, original_indices)
            active_count = active.sum(dim=1).to(torch.long)
            active = (
                torch.arange(c.dof, device=state.device).view(1, -1)
                < active_count.view(-1, 1)
            ).to(state.dtype)
        graph_obj = obj.detach() if self.detach_object_features else obj
        obj_nodes = graph_obj.unsqueeze(1).expand(-1, c.dof, -1)
        features = torch.cat(
            (q.unsqueeze(-1), qvel.unsqueeze(-1), action.unsqueeze(-1),
             depth.unsqueeze(-1), obj_nodes), dim=-1,
        )
        nodes = self.node_encoder(features) * active.unsqueeze(-1)
        for _ in range(c.message_steps):
            neighbors = (
                self._neighbor_sum_packed(nodes, active)
                if self.packed_active_nodes
                else self._compact_neighbor_sum(nodes, active)
            )
            message_inputs = [nodes, neighbors]
            if self.bridge_edge_features:
                message_inputs.append(self._compact_edge_features(active, lock_angle))
            messages = self.message(torch.cat(message_inputs, dim=-1))
            nodes = self.update(
                messages.reshape(-1, c.hidden_dim), nodes.reshape(-1, c.hidden_dim)
            ).view_as(nodes) * active.unsqueeze(-1)
        if hidden is None:
            hidden = torch.zeros_like(nodes)
        hidden = self.temporal(
            nodes.reshape(-1, c.hidden_dim), hidden.reshape(-1, c.hidden_dim)
        ).view_as(nodes) * active.unsqueeze(-1)
        delta = self.joint_head(hidden) * active.unsqueeze(-1)
        next_q = q + delta[..., 0]
        next_qvel = qvel + delta[..., 1]
        active_count = active.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden * active.unsqueeze(-1)).sum(dim=1) / active_count
        if self.detach_object_features:
            pooled = pooled.detach()
        next_obj = obj + self.object_head(torch.cat((pooled, obj), dim=-1))
        if original_indices is not None:
            full_q = torch.zeros_like(next_q).scatter(1, original_indices, next_q)
            full_qvel = torch.zeros_like(next_qvel).scatter(1, original_indices, next_qvel)
            next_q, next_qvel = full_q, full_qvel
        prediction = torch.cat((next_q, next_qvel, next_obj), dim=-1)
        return self.surgery.project_state(prediction, mask, lock_angle), hidden

    @staticmethod
    def _neighbor_sum_packed(nodes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        edge = active[:, 1:] * active[:, :-1]
        result[:, 1:] += nodes[:, :-1] * edge.unsqueeze(-1)
        result[:, :-1] += nodes[:, 1:] * edge.unsqueeze(-1)
        return result
