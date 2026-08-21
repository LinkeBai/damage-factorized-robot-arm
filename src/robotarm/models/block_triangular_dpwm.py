"""Block-triangular damage-projected world model (BT-DPWM).

The robot block is damage projected and autonomous with respect to the object.
The object block receives the projected robot representation through a directed,
stop-gradient bridge.  This makes the triangular dependency an executable
property of the model rather than a training convention.
"""
from __future__ import annotations

import torch
from torch import nn

from .topology_graph_world_model import TopologyGraphConfig
from .topology_surgery import TopologySurgery


class BlockTriangularDPWM(nn.Module):
    """Single world model with a robot -> object block-triangular transition."""

    def __init__(
        self, cfg: TopologyGraphConfig | None = None, *,
        contact_conditioned_robot: bool = False,
        independent_object_encoder: bool = False,
        object_hidden_dim: int | None = None,
        reaction_rank: int = 0,
        reaction_geometry_gate: bool = False,
        reaction_gate_threshold: float = -0.005,
        reaction_gate_temperature: float = 0.002,
    ) -> None:
        super().__init__()
        self.cfg = cfg or TopologyGraphConfig()
        self.contact_conditioned_robot = contact_conditioned_robot
        self.independent_object_encoder = independent_object_encoder
        c = self.cfg
        self.object_hidden_dim = object_hidden_dim or c.hidden_dim
        self.reaction_rank = reaction_rank
        self.reaction_geometry_gate = reaction_geometry_gate
        self.reaction_gate_threshold = reaction_gate_threshold
        self.reaction_gate_temperature = reaction_gate_temperature
        # q, qvel, projected action, locked, lock angle, normalized depth.
        # Y1 may additionally condition on the current object state.  This is a
        # forward contact context; the object loss still cannot cross the
        # detached robot -> object bridge and update the robot block.
        robot_input_dim = 6 + (c.object_dim if contact_conditioned_robot else 0)
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_input_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
        )
        self.robot_message = nn.Sequential(
            nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.robot_update = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.robot_temporal = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.robot_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, 2)
        )
        # Directed bridge: projected robot state/code -> object transition.
        self.object_head = nn.Sequential(
            nn.Linear(c.hidden_dim + 2 * c.dof + c.object_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.object_dim),
        )
        if independent_object_encoder:
            oh = self.object_hidden_dim
            # projected q/qvel, action, locked, lock angle, depth, object state
            object_input_dim = 6 + c.object_dim
            self.object_encoder = nn.Sequential(
                nn.Linear(object_input_dim, oh), nn.SiLU(),
                nn.Linear(oh, oh), nn.SiLU(),
            )
            self.object_message = nn.Sequential(
                nn.Linear(2 * oh, oh), nn.SiLU(),
                nn.Linear(oh, oh),
            )
            self.object_update = nn.GRUCell(oh, oh)
            self.object_temporal = nn.GRUCell(oh, oh)
            self.object_head = nn.Sequential(
                nn.Linear(oh + c.object_dim, oh), nn.SiLU(),
                nn.Linear(oh, c.object_dim),
            )
        self.surgery = TopologySurgery()
        if reaction_rank > 0:
            self.reaction_adapter = nn.Sequential(
                nn.Linear(c.hidden_dim + c.object_dim, reaction_rank), nn.Tanh(),
                nn.Linear(reaction_rank, 2),
            )
            nn.init.zeros_(self.reaction_adapter[-1].weight)
            nn.init.zeros_(self.reaction_adapter[-1].bias)
        if reaction_geometry_gate:
            self.register_buffer("reaction_axes", torch.tensor(
                [[0., 0., 1.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.]]
            ), persistent=False)
            self.register_buffer("reaction_origins", torch.tensor(
                [[0., 0., .120], [0., 0., 0.], [0., 0., .110],
                 [0., 0., .120], [0., 0., .060]]
            ), persistent=False)

    @staticmethod
    def _neighbor_sum(nodes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        result[:, 1:] += nodes[:, :-1]
        result[:, :-1] += nodes[:, 1:]
        return result

    @staticmethod
    def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / axis.norm().clamp_min(1e-12)
        x, y, z = axis
        zero = x * 0
        skew = torch.stack((torch.stack((zero, -z, y)),
                            torch.stack((z, zero, -x)),
                            torch.stack((-y, x, zero))))
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        outer = axis[:, None] * axis[None, :]
        return (torch.cos(angle)[..., None, None] * eye
                + (1.0 - torch.cos(angle))[..., None, None] * outer
                + torch.sin(angle)[..., None, None] * skew)

    def _reaction_contact_gate(self, q: torch.Tensor, block_xy: torch.Tensor) -> torch.Tensor:
        """Parameter-free soft gate from analytic pusher/box separation."""
        batch = q.shape[0]
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch, 3, 3).clone()
        position = torch.zeros(batch, 3, device=q.device, dtype=q.dtype)
        for joint in range(self.cfg.dof):
            origin = self.reaction_origins[joint].to(q).view(1, 3, 1).expand(batch, -1, -1)
            position = position + torch.bmm(rotation, origin).squeeze(-1)
            rotation = torch.bmm(rotation, self._axis_rotation(
                self.reaction_axes[joint].to(q), q[:, joint]
            ))
        tool_offset = q.new_tensor([0.0, -0.0132, 0.110])
        tool = position + torch.bmm(
            rotation, tool_offset.view(1, 3, 1).expand(batch, -1, -1)
        ).squeeze(-1)
        local = q.new_tensor([[0.0, 0.0, 0.0], [0.0200, 0.0, 0.0],
                              [0.0537, -0.0210, 0.0210]])
        points = torch.stack([
            tool + torch.bmm(rotation, point.view(1, 3, 1).expand(batch, -1, -1)).squeeze(-1)
            for point in local
        ], dim=1)[..., :2]
        gaps = []
        lower, upper = block_xy - 0.02, block_xy + 0.02
        for index, radius in ((0, 0.012), (1, 0.008)):
            first, direction = points[:, index], points[:, index + 1] - points[:, index]
            safe = torch.where(direction.abs() > 1e-8, direction, torch.ones_like(direction))
            fractions = torch.stack((
                torch.zeros(batch, device=q.device, dtype=q.dtype),
                torch.ones(batch, device=q.device, dtype=q.dtype),
                (lower[:, 0] - first[:, 0]) / safe[:, 0],
                (upper[:, 0] - first[:, 0]) / safe[:, 0],
                (lower[:, 1] - first[:, 1]) / safe[:, 1],
                (upper[:, 1] - first[:, 1]) / safe[:, 1],
                ((block_xy - first) * direction).sum(-1) / direction.pow(2).sum(-1).clamp_min(1e-8),
            ), dim=1).clamp(0.0, 1.0)
            segment_points = first[:, None, :] + fractions[..., None] * direction[:, None, :]
            box_points = torch.maximum(torch.minimum(segment_points, upper[:, None, :]),
                                       lower[:, None, :])
            distance = torch.linalg.vector_norm(box_points - segment_points, dim=-1).min(dim=1).values
            gaps.append(distance - radius)
        gap = torch.stack(gaps, dim=1).min(dim=1).values
        return torch.sigmoid((self.reaction_gate_threshold - gap)
                             / self.reaction_gate_temperature)

    def step(self, state, action, mask, lock_angle, hidden):
        object_hidden = None
        if self.independent_object_encoder and hidden is not None:
            object_hidden = hidden[1] if isinstance(hidden, tuple) else hidden[:, self.cfg.dof:]
        projected_robot, next_hidden, obj, action, depth = self.step_robot(
            state, action, mask, lock_angle, hidden
        )
        return self.step_object(
            projected_robot, obj, action, mask, lock_angle, depth, next_hidden,
            object_hidden,
        )

    def step_robot(self, state, action, mask, lock_angle, hidden):
        """Advance only the robot block, skipping all object-block compute."""
        c = self.cfg
        state = self.surgery.project_state(state, mask, lock_angle)
        action = self.surgery.project_action(action, mask)
        q, qvel, obj = state[:, :c.dof], state[:, c.dof:2*c.dof], state[:, 2*c.dof:]
        depth = torch.linspace(0.0, 1.0, c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, -1).expand(state.shape[0], -1)
        features = torch.stack((q, qvel, action, mask, lock_angle, depth), dim=-1)
        if self.contact_conditioned_robot:
            object_context = obj.unsqueeze(1).expand(-1, c.dof, -1)
            features = torch.cat((features, object_context), dim=-1)
        nodes = self.robot_encoder(features)
        for _ in range(c.message_steps):
            messages = self.robot_message(torch.cat((nodes, self._neighbor_sum(nodes)), -1))
            nodes = self.robot_update(messages.flatten(0, 1), nodes.flatten(0, 1)).view_as(nodes)
        robot_hidden = hidden
        if self.independent_object_encoder and hidden is not None:
            robot_hidden = hidden[0] if isinstance(hidden, tuple) else hidden[:, :c.dof]
        if robot_hidden is None:
            robot_hidden = torch.zeros_like(nodes)
        next_hidden = self.robot_temporal(
            nodes.flatten(0, 1), robot_hidden.flatten(0, 1)
        ).view_as(nodes)
        delta = self.robot_head(next_hidden)
        robot = torch.cat((q + delta[..., 0], qvel + delta[..., 1]), -1)
        provisional = torch.cat((robot, obj), -1)
        projected_robot = self.surgery.project_state(provisional, mask, lock_angle)[:, :2*c.dof]
        if self.reaction_rank > 0:
            context = obj.unsqueeze(1).expand(-1, c.dof, -1)
            reaction = self.reaction_adapter(torch.cat((next_hidden, context), -1))
            if self.reaction_geometry_gate:
                reaction = reaction * self._reaction_contact_gate(q, obj[:, :2]).view(-1, 1, 1)
            reaction = reaction * (1.0 - mask).unsqueeze(-1)
            corrected = projected_robot.clone()
            corrected[:, :c.dof] += reaction[..., 0]
            corrected[:, c.dof:] += reaction[..., 1]
            projected_robot = self.surgery.project_state(
                torch.cat((corrected, obj), -1), mask, lock_angle)[:, :2*c.dof]
        return projected_robot, next_hidden, obj, action, depth

    def step_object(
        self, projected_robot, obj, action, mask, lock_angle, depth, robot_hidden,
        object_hidden=None,
    ):
        """Advance the object block from a projected robot transition."""
        c = self.cfg
        if self.independent_object_encoder:
            projected_q = projected_robot[:, :c.dof]
            projected_qvel = projected_robot[:, c.dof:]
            object_nodes = obj.unsqueeze(1).expand(-1, c.dof, -1)
            object_features = torch.cat((
                projected_q.unsqueeze(-1), projected_qvel.unsqueeze(-1),
                action.unsqueeze(-1), mask.unsqueeze(-1), lock_angle.unsqueeze(-1),
                depth.unsqueeze(-1), object_nodes,
            ), -1).detach()
            object_code = self.object_encoder(object_features)
            for _ in range(c.message_steps):
                messages = self.object_message(torch.cat(
                    (object_code, self._neighbor_sum(object_code)), -1))
                object_code = self.object_update(
                    messages.flatten(0, 1), object_code.flatten(0, 1)
                ).view_as(object_code)
            if object_hidden is None:
                object_hidden = torch.zeros_like(object_code)
            next_object_hidden = self.object_temporal(
                object_code.flatten(0, 1), object_hidden.flatten(0, 1)
            ).view_as(object_code)
            next_obj = obj + self.object_head(torch.cat(
                (next_object_hidden.mean(1), obj), -1))
            returned_hidden = (
                torch.cat((robot_hidden, next_object_hidden), 1)
                if robot_hidden.shape[-1] == next_object_hidden.shape[-1]
                else (robot_hidden, next_object_hidden)
            )
        else:
            bridge = torch.cat((robot_hidden.mean(1), projected_robot), -1).detach()
            next_obj = obj + self.object_head(torch.cat((bridge, obj), -1))
            returned_hidden = robot_hidden
        prediction = torch.cat((projected_robot, next_obj), -1)
        return self.surgery.project_state(prediction, mask, lock_angle), returned_hidden
