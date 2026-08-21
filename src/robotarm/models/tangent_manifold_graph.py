"""Damage-conditioned tangent dynamics on the full serial-chain graph."""
from __future__ import annotations

import torch

from .topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel


class TangentManifoldGraphWorldModel(TopologyGraphWorldModel):
    """Keep spatial relay nodes but constrain temporal dynamics to free joints.

    Locked nodes remain in the per-step message-passing graph, so the serial
    chain is not contracted. Their recurrent memory and predicted increments
    are removed analytically, implementing a tangent/normal decomposition for
    joint-lock damage.
    """

    def __init__(self, cfg: TopologyGraphConfig | None = None) -> None:
        super().__init__(cfg)

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
        active = (1.0 - mask).to(state)
        q, qvel = state[:, :c.dof], state[:, c.dof:2 * c.dof]
        obj = state[:, 2 * c.dof:]
        depth = torch.linspace(0.0, 1.0, c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, c.dof).expand(state.shape[0], -1)
        features = torch.cat((
            q.unsqueeze(-1), qvel.unsqueeze(-1), action.unsqueeze(-1),
            mask.unsqueeze(-1), lock_angle.unsqueeze(-1), depth.unsqueeze(-1),
            obj.unsqueeze(1).expand(-1, c.dof, -1),
        ), dim=-1)
        # Locked nodes participate in spatial messages as geometry relays.
        nodes = self.node_encoder(features)
        for _ in range(c.message_steps):
            messages = self.message(torch.cat((nodes, self._neighbor_sum(nodes)), dim=-1))
            nodes = self.update(
                messages.reshape(-1, c.hidden_dim), nodes.reshape(-1, c.hidden_dim)
            ).view_as(nodes)
        if hidden is None:
            hidden = torch.zeros_like(nodes)
        hidden = hidden * active.unsqueeze(-1)
        hidden = self.temporal(
            nodes.reshape(-1, c.hidden_dim), hidden.reshape(-1, c.hidden_dim)
        ).view_as(nodes)
        # Temporal state and learned motion live only in T(M_d).
        hidden = hidden * active.unsqueeze(-1)
        delta = self.joint_head(hidden) * active.unsqueeze(-1)
        prediction = state.clone()
        prediction[:, :c.dof] = q + delta[..., 0]
        prediction[:, c.dof:2 * c.dof] = qvel + delta[..., 1]
        return self.surgery.project_state(prediction, mask, lock_angle), hidden
