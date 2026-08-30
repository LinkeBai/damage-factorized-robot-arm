"""Event-driven, friction-constrained object dynamics for planar pushing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fixed_transform_graph import FixedTransformGraphConfig, FixedTransformGraphWorldModel


@dataclass
class HybridContactConfig:
    dof: int = 5
    hidden_dim: int = 64
    time_step: float = 0.005
    friction_coefficient: float = 0.8
    contact_radius: float = 0.028


class HybridContactImpulseModel(nn.Module):
    """Predict a 2-D contact impulse and apply it through analytic integration.

    Damage cannot be ignored: pusher geometry is computed through the complete
    fixed-transform chain.  The oracle contact mask is deliberately an explicit
    input for Gate M2; a deployable geometric mode detector belongs to Gate M3.
    """

    def __init__(self, cfg: HybridContactConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HybridContactConfig()
        geometry_cfg = FixedTransformGraphConfig(dof=self.cfg.dof, hidden_dim=16)
        self.geometry = FixedTransformGraphWorldModel(geometry_cfg)
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(False)
        # gap, normal(2), tangent(2), pusher velocity(2), object velocity(2),
        # relative normal/tangent velocity, current/next segment endpoints(8)
        self.impulse_net = nn.Sequential(
            nn.Linear(19, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, 2),
        )
        # Positive decay rate gives a stable analytic free-object transition.
        self.raw_drag = nn.Parameter(torch.tensor(-2.0))

    def _pusher_endpoints_xy(self, q: torch.Tensor) -> torch.Tensor:
        positions, rotations = self.geometry._joint_poses(q)
        wrist_position, wrist_rotation = positions[:, -1], rotations[:, -1]
        tool_offset = q.new_tensor([0.0, -0.0132, 0.110])
        local_endpoints = q.new_tensor(
            [[0.0200, 0.0, 0.0], [0.0537, -0.0210, 0.0210]]
        )
        tool_position = wrist_position + torch.bmm(
            wrist_rotation, tool_offset.view(1, 3, 1).expand(q.shape[0], -1, -1)
        ).squeeze(-1)
        return torch.stack(
            [
                tool_position
                + torch.bmm(
                    wrist_rotation,
                    endpoint.view(1, 3, 1).expand(q.shape[0], -1, -1),
                ).squeeze(-1)
                for endpoint in local_endpoints
            ],
            dim=1,
        )[..., :2]

    def contact_features(
        self, state: torch.Tensor, next_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj = state[:, 2 * self.cfg.dof:]
        block_xy, object_velocity = obj[:, :2], obj[:, 2:]
        current = self._pusher_endpoints_xy(state[:, :self.cfg.dof])
        following = self._pusher_endpoints_xy(next_q)
        segment = current[:, 1] - current[:, 0]
        fraction = (
            ((block_xy - current[:, 0]) * segment).sum(-1)
            / segment.pow(2).sum(-1).clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        closest = current[:, 0] + fraction[:, None] * segment
        relative = block_xy - closest
        distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1e-6)
        normal = relative / distance[:, None]
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=-1)
        next_closest = following[:, 0] + fraction[:, None] * (
            following[:, 1] - following[:, 0]
        )
        pusher_velocity = (next_closest - closest) / self.cfg.time_step
        relative_velocity = pusher_velocity - object_velocity
        normal_speed = (relative_velocity * normal).sum(-1, keepdim=True)
        tangent_speed = (relative_velocity * tangent).sum(-1, keepdim=True)
        gap = distance[:, None] - self.cfg.contact_radius
        features = torch.cat(
            (
                gap,
                normal,
                tangent,
                pusher_velocity,
                object_velocity,
                normal_speed,
                tangent_speed,
                current.flatten(1),
                following.flatten(1),
            ),
            dim=-1,
        )
        return features, normal, tangent

    def candidate_contact_frames(
        self, state: torch.Tensor, next_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool and pusher capsule candidates in the table plane.

        Returns gaps, normals and tangents with candidate order ``tool,pusher``.
        The oracle Gate N supplies candidate activity; learning activity is a
        separate gate.
        """
        obj = state[:, 2 * self.cfg.dof:]
        block_xy = obj[:, :2]
        positions, rotations = self.geometry._joint_poses(state[:, :self.cfg.dof])
        wrist_position, wrist_rotation = positions[:, -1], rotations[:, -1]
        tool_offset = state.new_tensor([0.0, -0.0132, 0.110])
        tool_position = wrist_position + torch.bmm(
            wrist_rotation,
            tool_offset.view(1, 3, 1).expand(state.shape[0], -1, -1),
        ).squeeze(-1)
        local_points = state.new_tensor(
            [[0.0, 0.0, 0.0], [0.0200, 0.0, 0.0], [0.0537, -0.0210, 0.0210]]
        )
        points = torch.stack(
            [
                tool_position + torch.bmm(
                    wrist_rotation,
                    point.view(1, 3, 1).expand(state.shape[0], -1, -1),
                ).squeeze(-1)
                for point in local_points
            ],
            dim=1,
        )[..., :2]
        normals, gaps = [], []
        for index, radius in ((0, 0.012), (1, 0.008)):
            first, second = points[:, index], points[:, index + 1]
            segment = second - first
            fraction = (
                ((block_xy - first) * segment).sum(-1)
                / segment.pow(2).sum(-1).clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            closest = first + fraction[:, None] * segment
            relative = block_xy - closest
            distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1e-6)
            normals.append(relative / distance[:, None])
            # Approximate block by a bounding circle only for candidate gating;
            # Gate N0 evaluates whether the resulting basis is expressive.
            gaps.append(distance - (0.02 * 2 ** 0.5 + radius))
        normal = torch.stack(normals, dim=1)
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        return torch.stack(gaps, dim=1), normal, tangent

    @staticmethod
    def _segment_aabb_frame(
        first: torch.Tensor,
        second: torch.Tensor,
        center: torch.Tensor,
        half_extent: float = 0.02,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact planar segment/AABB closest-point frame for axis-aligned blocks."""
        direction = second - first
        safe_x = torch.where(
            direction[:, 0].abs() > 1e-8, direction[:, 0], torch.ones_like(direction[:, 0])
        )
        safe_y = torch.where(
            direction[:, 1].abs() > 1e-8, direction[:, 1], torch.ones_like(direction[:, 1])
        )
        candidates = [
            torch.zeros(first.shape[0], device=first.device, dtype=first.dtype),
            torch.ones(first.shape[0], device=first.device, dtype=first.dtype),
            (((center[:, 0] - half_extent) - first[:, 0]) / safe_x),
            (((center[:, 0] + half_extent) - first[:, 0]) / safe_x),
            (((center[:, 1] - half_extent) - first[:, 1]) / safe_y),
            (((center[:, 1] + half_extent) - first[:, 1]) / safe_y),
            ((center - first) * direction).sum(-1)
            / direction.pow(2).sum(-1).clamp_min(1e-8),
        ]
        fraction = torch.stack(candidates, dim=1).clamp(0.0, 1.0)
        segment_points = first[:, None, :] + fraction[..., None] * direction[:, None, :]
        lower, upper = center - half_extent, center + half_extent
        box_points = torch.maximum(
            torch.minimum(segment_points, upper[:, None, :]), lower[:, None, :]
        )
        difference = box_points - segment_points
        squared = difference.pow(2).sum(-1)
        index = squared.argmin(dim=1)
        batch = torch.arange(first.shape[0], device=first.device)
        best = difference[batch, index]
        distance = torch.linalg.vector_norm(best, dim=-1)
        fallback = center - segment_points[batch, index]
        fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(1e-8)
        normal = torch.where(
            (distance > 1e-7)[:, None], best / distance.clamp_min(1e-8)[:, None], fallback
        )
        return distance, normal

    def candidate_box_contact_frames(
        self, state: torch.Tensor, next_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool/pusher capsule candidates against the actual axis-aligned box."""
        block_xy = state[:, 2 * self.cfg.dof:2 * self.cfg.dof + 2]
        positions, rotations = self.geometry._joint_poses(state[:, :self.cfg.dof])
        wrist_position, wrist_rotation = positions[:, -1], rotations[:, -1]
        tool_offset = state.new_tensor([0.0, -0.0132, 0.110])
        tool_position = wrist_position + torch.bmm(
            wrist_rotation,
            tool_offset.view(1, 3, 1).expand(state.shape[0], -1, -1),
        ).squeeze(-1)
        local_points = state.new_tensor(
            [[0.0, 0.0, 0.0], [0.0200, 0.0, 0.0], [0.0537, -0.0210, 0.0210]]
        )
        points = torch.stack(
            [
                tool_position + torch.bmm(
                    wrist_rotation,
                    point.view(1, 3, 1).expand(state.shape[0], -1, -1),
                ).squeeze(-1)
                for point in local_points
            ],
            dim=1,
        )[..., :2]
        distances, normals = [], []
        for index, radius in ((0, 0.012), (1, 0.008)):
            distance, normal = self._segment_aabb_frame(
                points[:, index], points[:, index + 1], block_xy
            )
            distances.append(distance - radius)
            normals.append(normal)
        normal = torch.stack(normals, dim=1)
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        return torch.stack(distances, dim=1), normal, tangent

    def forward(
        self, state: torch.Tensor, next_q: torch.Tensor, contact_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, normal, tangent = self.contact_features(state, next_q)
        raw = self.impulse_net(features)
        normal_impulse = torch.nn.functional.softplus(raw[:, :1])
        tangent_impulse = (
            self.cfg.friction_coefficient
            * normal_impulse
            * torch.tanh(raw[:, 1:2])
        )
        active = contact_mask.to(state.dtype).view(-1, 1)
        delta_velocity = active * (
            normal_impulse * normal + tangent_impulse * tangent
        )
        obj = state[:, 2 * self.cfg.dof:]
        drag = torch.nn.functional.softplus(self.raw_drag)
        free_velocity = obj[:, 2:] * torch.exp(-drag * self.cfg.time_step)
        next_velocity = free_velocity + delta_velocity
        next_position = obj[:, :2] + self.cfg.time_step * next_velocity
        prediction = torch.cat((next_position, next_velocity), dim=-1)
        diagnostics = {
            "normal_impulse": normal_impulse,
            "tangent_impulse": tangent_impulse,
            "delta_velocity": delta_velocity,
            "features": features,
        }
        return prediction, diagnostics


def oracle_velocity_impulse(
    state: torch.Tensor, next_state: torch.Tensor, *, dof: int = 5,
    time_step: float = 0.005, drag_rate: float = 0.0,
) -> torch.Tensor:
    """Velocity impulse required after the analytic free transition."""
    velocity = state[:, 2 * dof + 2:2 * dof + 4]
    target_velocity = next_state[:, 2 * dof + 2:2 * dof + 4]
    free_velocity = velocity * torch.exp(velocity.new_tensor(-drag_rate * time_step))
    return target_velocity - free_velocity
