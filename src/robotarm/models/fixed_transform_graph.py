"""Geometry-preserving graph dynamics for known locked joints."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .topology_surgery import TopologySurgery


@dataclass
class FixedTransformGraphConfig:
    dof: int = 5
    hidden_dim: int = 128
    message_steps: int = 2


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues rotation for shared local axes."""
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis
    skew = torch.stack((
        torch.stack((x * 0, -z, y)),
        torch.stack((z, y * 0, -x)),
        torch.stack((-y, x, z * 0)),
    ))
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis[:, None] * axis[None, :]
    return (
        torch.cos(angle)[..., None, None] * eye
        + (1.0 - torch.cos(angle))[..., None, None] * outer
        + torch.sin(angle)[..., None, None] * skew
    )


class FixedTransformGraphWorldModel(nn.Module):
    """Keep locked links as fixed SE(3) message relays; predict free joints only."""

    def __init__(self, cfg: FixedTransformGraphConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FixedTransformGraphConfig()
        c = self.cfg
        self.register_buffer("axes", torch.tensor(
            [[0., 0., 1.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.]]
        ))
        self.register_buffer("origins", torch.tensor(
            [[0., 0., .120], [0., 0., 0.], [0., 0., .110],
             [0., 0., .120], [0., 0., .060]]
        ))
        # q, qvel, action, locked, depth, world position(3), rotation-6D
        self.node_encoder = nn.Sequential(
            nn.Linear(14, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
        )
        # source hidden + relative translation(3) + relative rotation-6D + fixed relay flag
        self.edge_message = nn.Sequential(
            nn.Linear(c.hidden_dim + 10, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.update = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.temporal = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.joint_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, 2)
        )
        self.surgery = TopologySurgery()

    def _joint_poses(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, dof = q.shape
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch, 3, 3).clone()
        position = torch.zeros(batch, 3, device=q.device, dtype=q.dtype)
        positions, rotations = [], []
        for joint in range(dof):
            position = position + torch.bmm(
                rotation, self.origins[joint].to(q).view(1, 3, 1).expand(batch, -1, -1)
            ).squeeze(-1)
            rotation = torch.bmm(
                rotation, _axis_rotation(self.axes[joint].to(q), q[:, joint])
            )
            positions.append(position)
            rotations.append(rotation)
        return torch.stack(positions, dim=1), torch.stack(rotations, dim=1)

    @staticmethod
    def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
        return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)

    def _messages(
        self, nodes: torch.Tensor, positions: torch.Tensor,
        rotations: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 4)):
            for source, target in ((left, right), (right, left)):
                target_r_t = rotations[:, target].transpose(-1, -2)
                relative_position = torch.bmm(
                    target_r_t, (positions[:, source] - positions[:, target]).unsqueeze(-1)
                ).squeeze(-1)
                relative_rotation = torch.bmm(target_r_t, rotations[:, source])
                fixed_relay = mask[:, source].unsqueeze(-1)
                edge = torch.cat(
                    (relative_position, self._rotation_6d(relative_rotation), fixed_relay), dim=-1
                )
                result[:, target] += self.edge_message(torch.cat((nodes[:, source], edge), dim=-1))
        return result

    def step(
        self, state: torch.Tensor, action: torch.Tensor,
        mask: torch.Tensor, lock_angle: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        state = self.surgery.project_state(state, mask, lock_angle)
        action = self.surgery.project_action(action, mask)
        q, qvel = state[:, :c.dof], state[:, c.dof:2 * c.dof]
        positions, rotations = self._joint_poses(q)
        depth = torch.linspace(0., 1., c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, -1).expand(state.shape[0], -1)
        features = torch.cat((
            q.unsqueeze(-1), qvel.unsqueeze(-1), action.unsqueeze(-1),
            mask.unsqueeze(-1), depth.unsqueeze(-1), positions,
            self._rotation_6d(rotations),
        ), dim=-1)
        nodes = self.node_encoder(features)
        for _ in range(c.message_steps):
            messages = self._messages(nodes, positions, rotations, mask)
            nodes = self.update(
                messages.reshape(-1, c.hidden_dim), nodes.reshape(-1, c.hidden_dim)
            ).view_as(nodes)
        if hidden is None:
            hidden = torch.zeros_like(nodes)
        hidden = self.temporal(
            nodes.reshape(-1, c.hidden_dim), hidden.reshape(-1, c.hidden_dim)
        ).view_as(nodes)
        delta = self.joint_head(hidden) * (1.0 - mask).unsqueeze(-1)
        prediction = state.clone()
        prediction[:, :c.dof] = q + delta[..., 0]
        prediction[:, c.dof:2 * c.dof] = qvel + delta[..., 1]
        return self.surgery.project_state(prediction, mask, lock_angle), hidden


class FixedTransformGraphObjectWorldModel(FixedTransformGraphWorldModel):
    """K1 joint transition plus an isolated low-capacity object residual head."""

    def __init__(
        self, cfg: FixedTransformGraphConfig | None = None, bottleneck_dim: int = 16
    ) -> None:
        super().__init__(cfg)
        c = self.cfg
        # Detached pooled joint state + detached end-effector SE(3) + object state.
        self.object_head = nn.Sequential(
            nn.Linear(c.hidden_dim + 9 + 4, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 4),
        )

    def step(
        self, state: torch.Tensor, action: torch.Tensor,
        mask: torch.Tensor, lock_angle: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction, next_hidden = super().step(state, action, mask, lock_angle, hidden)
        positions, rotations = self._joint_poses(prediction[:, :self.cfg.dof])
        geometry = torch.cat(
            (positions[:, -1], self._rotation_6d(rotations[:, -1])), dim=-1
        ).detach()
        object_state = state[:, 2 * self.cfg.dof:]
        object_input = torch.cat(
            (next_hidden.mean(dim=1).detach(), geometry, object_state), dim=-1
        )
        prediction = prediction.clone()
        prediction[:, 2 * self.cfg.dof:] = object_state + self.object_head(object_input)
        return prediction, next_hidden


class FixedTransformContactWorldModel(FixedTransformGraphWorldModel):
    """Fixed-transform dynamics with factorized contact reaction and impulse."""

    def __init__(
        self, cfg: FixedTransformGraphConfig | None = None,
        contact_dim: int = 32, bottleneck_dim: int = 16,
    ) -> None:
        super().__init__(cfg)
        c = self.cfg
        # object(4), current endpoints(4), endpoint motion(4), two relative
        # vectors(4), two signed gaps(2), and two soft contact gates(2).
        contact_input_dim = 20
        self.joint_contact_encoder = nn.Sequential(
            nn.Linear(contact_input_dim, contact_dim), nn.SiLU(),
            nn.Linear(contact_dim, contact_dim), nn.SiLU(),
        )
        self.joint_reaction_head = nn.Sequential(
            nn.Linear(c.hidden_dim + contact_dim, bottleneck_dim), nn.SiLU(),
            nn.Linear(bottleneck_dim, 2),
        )
        self.object_contact_encoder = nn.Sequential(
            nn.Linear(contact_input_dim, contact_dim), nn.SiLU(),
            nn.Linear(contact_dim, contact_dim), nn.SiLU(),
        )
        self.object_free_head = nn.Sequential(
            nn.Linear(4, bottleneck_dim), nn.SiLU(), nn.Linear(bottleneck_dim, 4)
        )
        self.object_impulse_head = nn.Sequential(
            nn.Linear(contact_dim, bottleneck_dim), nn.SiLU(), nn.Linear(bottleneck_dim, 4)
        )
        self.time_step = 0.005

    def _pusher_endpoints_xy(self, q: torch.Tensor) -> torch.Tensor:
        positions, rotations = self._joint_poses(q)
        wrist_position, wrist_rotation = positions[:, -1], rotations[:, -1]
        tool_offset = q.new_tensor([0.0, -0.0132, 0.110])
        local_endpoints = q.new_tensor(
            [[0.0200, 0.0, 0.0], [0.0537, -0.0210, 0.0210]]
        )
        tool_position = wrist_position + torch.bmm(
            wrist_rotation, tool_offset.view(1, 3, 1).expand(q.shape[0], -1, -1)
        ).squeeze(-1)
        endpoints = torch.stack(
            [
                tool_position + torch.bmm(
                    wrist_rotation,
                    endpoint.view(1, 3, 1).expand(q.shape[0], -1, -1),
                ).squeeze(-1)
                for endpoint in local_endpoints
            ],
            dim=1,
        )
        return endpoints[..., :2]

    @staticmethod
    def _relative_to_segment(
        endpoints: torch.Tensor, block_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second = endpoints[:, 0], endpoints[:, 1]
        segment = second - first
        fraction = (
            ((block_xy - first) * segment).sum(dim=-1)
            / segment.pow(2).sum(dim=-1).clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        closest = first + fraction.unsqueeze(-1) * segment
        relative = block_xy - closest
        signed_gap = torch.linalg.vector_norm(relative, dim=-1) - 0.028
        gate = torch.sigmoid(-signed_gap / 0.01)
        return relative, signed_gap, gate

    def _contact_features(
        self, state: torch.Tensor, provisional_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_state = state[:, 2 * self.cfg.dof:]
        block_xy = object_state[:, :2]
        current = self._pusher_endpoints_xy(state[:, :self.cfg.dof])
        provisional = self._pusher_endpoints_xy(provisional_q)
        current_relative, current_gap, current_gate = self._relative_to_segment(
            current, block_xy
        )
        next_relative, next_gap, next_gate = self._relative_to_segment(
            provisional, block_xy
        )
        features = torch.cat(
            (
                object_state, current.flatten(1), (provisional - current).flatten(1),
                current_relative, next_relative, current_gap.unsqueeze(-1),
                next_gap.unsqueeze(-1), current_gate.unsqueeze(-1),
                next_gate.unsqueeze(-1),
            ),
            dim=-1,
        )
        return features, torch.maximum(current_gate, next_gate)

    def step(
        self, state: torch.Tensor, action: torch.Tensor,
        mask: torch.Tensor, lock_angle: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction, next_hidden = super().step(state, action, mask, lock_angle, hidden)
        features, contact_gate = self._contact_features(state, prediction[:, :self.cfg.dof])

        joint_code = self.joint_contact_encoder(features.detach())
        repeated_code = joint_code.unsqueeze(1).expand(-1, self.cfg.dof, -1)
        reaction = self.joint_reaction_head(
            torch.cat((next_hidden, repeated_code), dim=-1)
        )
        reaction = reaction * contact_gate[:, None, None]
        reaction = reaction * (1.0 - mask).unsqueeze(-1)
        prediction = prediction.clone()
        prediction[:, :self.cfg.dof] += reaction[..., 0]
        prediction[:, self.cfg.dof:2 * self.cfg.dof] += reaction[..., 1]
        prediction = self.surgery.project_state(prediction, mask, lock_angle)

        object_state = state[:, 2 * self.cfg.dof:]
        inertial = torch.cat(
            (object_state[:, :2] + self.time_step * object_state[:, 2:], object_state[:, 2:]),
            dim=-1,
        )
        object_code = self.object_contact_encoder(features.detach())
        object_delta = self.object_free_head(object_state)
        object_delta = object_delta + contact_gate.detach().unsqueeze(-1) * self.object_impulse_head(
            object_code
        )
        prediction[:, 2 * self.cfg.dof:] = inertial + object_delta
        return prediction, next_hidden
