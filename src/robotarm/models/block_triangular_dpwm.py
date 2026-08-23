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
from .contact_geometry import pusher_box_contact_gate, pusher_reference_point


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
        reaction_scale: float = 1.0,
        reaction_physical_features: bool = False,
        reaction_event_decay: float | None = None,
        reaction_fixed_initialization: bool = False,
        kinematic_integration_dt: float | None = None,
        kinematic_position_blend: float = 1.0,
        shadow_object_rank: int = 0,
        robot_expert_count: int = 1,
        contact_gated_object_context: bool = False,
        linear_physical_reaction: bool = False,
        robot_position_delta_scale: float = 1.0,
        robot_velocity_delta_scale: float = 1.0,
        reaction_relative_clip: float | None = None,
        compact_bridge_object_head: bool = False,
        geometric_object_rank: int = 0,
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
        self.reaction_scale = reaction_scale
        self.reaction_physical_features = reaction_physical_features
        self.reaction_event_decay = reaction_event_decay
        self.reaction_fixed_initialization = reaction_fixed_initialization
        self.kinematic_integration_dt = kinematic_integration_dt
        self.kinematic_position_blend = kinematic_position_blend
        self.shadow_object_rank = shadow_object_rank
        self.robot_expert_count = robot_expert_count
        self.contact_gated_object_context = contact_gated_object_context
        self.linear_physical_reaction = linear_physical_reaction
        self.robot_position_delta_scale = robot_position_delta_scale
        self.robot_velocity_delta_scale = robot_velocity_delta_scale
        self.reaction_relative_clip = reaction_relative_clip
        self.compact_bridge_object_head = compact_bridge_object_head
        self.geometric_object_rank = int(geometric_object_rank)
        if compact_bridge_object_head and independent_object_encoder:
            raise ValueError("compact bridge and independent object encoder are exclusive")
        if reaction_relative_clip is not None and reaction_relative_clip < 0:
            raise ValueError("reaction_relative_clip must be non-negative")
        if robot_expert_count < 1:
            raise ValueError("robot_expert_count must be positive")
        if geometric_object_rank < 0:
            raise ValueError("geometric_object_rank must be non-negative")
        if robot_expert_count > 1 and (reaction_rank > 0 or shadow_object_rank > 0):
            raise ValueError("robot ensembles cannot combine with reaction or shadow in this gate")
        if shadow_object_rank > 0 and reaction_event_decay is not None:
            raise ValueError("shadow object context and reaction event trace cannot share hidden slot")
        if not 0.0 <= kinematic_position_blend <= 1.0:
            raise ValueError("kinematic_position_blend must be in [0, 1]")
        if reaction_event_decay is not None and not 0.0 <= reaction_event_decay < 1.0:
            raise ValueError("reaction_event_decay must be in [0, 1)")
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
        self.additional_robot_experts = nn.ModuleList()
        for _ in range(robot_expert_count - 1):
            self.additional_robot_experts.append(nn.ModuleDict({
                "encoder": nn.Sequential(
                    nn.Linear(robot_input_dim, c.hidden_dim), nn.SiLU(),
                    nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
                ),
                "message": nn.Sequential(
                    nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.SiLU(),
                    nn.Linear(c.hidden_dim, c.hidden_dim),
                ),
                "updater": nn.GRUCell(c.hidden_dim, c.hidden_dim),
                "temporal": nn.GRUCell(c.hidden_dim, c.hidden_dim),
                "head": nn.Sequential(
                    nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
                    nn.Linear(c.hidden_dim, 2),
                ),
            }))
        # Directed bridge: projected robot state/code -> object transition.
        self.object_head = nn.Sequential(
            nn.Linear(c.hidden_dim + 2 * c.dof + c.object_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.object_dim),
        )
        if compact_bridge_object_head:
            self.object_head = nn.Sequential(
                nn.Linear(c.hidden_dim + c.object_dim, c.hidden_dim), nn.SiLU(),
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
            reaction_input_dim = (robot_input_dim if reaction_physical_features
                                  else c.hidden_dim + c.object_dim)
            self.reaction_adapter = nn.Sequential(
                nn.Linear(reaction_input_dim, reaction_rank), nn.Tanh(),
                nn.Linear(reaction_rank, 2),
            )
            if reaction_fixed_initialization:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(314159)
                    self.reaction_adapter[0].reset_parameters()
            nn.init.zeros_(self.reaction_adapter[-1].weight)
            nn.init.zeros_(self.reaction_adapter[-1].bias)
        if linear_physical_reaction:
            self.linear_reaction_adapter = nn.Linear(robot_input_dim, 2)
            nn.init.zeros_(self.linear_reaction_adapter.weight)
            nn.init.zeros_(self.linear_reaction_adapter.bias)
        if reaction_geometry_gate:
            self.register_buffer("reaction_axes", torch.tensor(
                [[0., 0., 1.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.]]
            ), persistent=False)
            self.register_buffer("reaction_origins", torch.tensor(
                [[0., 0., .120], [0., 0., 0.], [0., 0., .110],
                 [0., 0., .120], [0., 0., .060]]
            ), persistent=False)
        if shadow_object_rank > 0:
            self.shadow_context_head = nn.Sequential(
                nn.Linear(c.hidden_dim + c.object_dim, shadow_object_rank), nn.Tanh(),
                nn.Linear(shadow_object_rank, c.object_dim),
            )
            nn.init.zeros_(self.shadow_context_head[-1].weight)
            nn.init.zeros_(self.shadow_context_head[-1].bias)
        if geometric_object_rank > 0:
            # Previous/next pusher xy, pusher displacement, relative pusher-box
            # xy, object velocity, and previous/next analytic contact gates.
            self.geometric_object_head = nn.Sequential(
                nn.Linear(12, geometric_object_rank), nn.Tanh(),
                nn.Linear(geometric_object_rank, c.object_dim),
            )
            nn.init.zeros_(self.geometric_object_head[-1].weight)
            nn.init.zeros_(self.geometric_object_head[-1].bias)

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
        shadow_obj = None
        if self.shadow_object_rank > 0:
            shadow_obj = (hidden[2] if isinstance(hidden, tuple) and len(hidden) == 3
                          else state[:, 2 * self.cfg.dof:])
        projected_robot, next_hidden, obj, action, depth = self.step_robot(
            state, action, mask, lock_angle, hidden, robot_context=shadow_obj
        )
        reaction_trace = None
        if self.reaction_event_decay is not None:
            next_hidden, reaction_trace = next_hidden
        prediction, returned_hidden = self.step_object(
            projected_robot, obj, action, mask, lock_angle, depth, next_hidden,
            object_hidden, previous_robot=state[:, :2 * self.cfg.dof],
        )
        if shadow_obj is not None:
            next_shadow = shadow_obj + self.shadow_context_head(torch.cat(
                (next_hidden.mean(1), shadow_obj), -1
            ))
            if isinstance(returned_hidden, tuple):
                returned_hidden = (*returned_hidden, next_shadow)
            else:
                returned_hidden = (returned_hidden, next_shadow)
        if reaction_trace is not None:
            if isinstance(returned_hidden, tuple):
                returned_hidden = (*returned_hidden, reaction_trace)
            else:
                returned_hidden = (returned_hidden, reaction_trace)
        return prediction, returned_hidden

    def step_robot(self, state, action, mask, lock_angle, hidden, robot_context=None):
        """Advance only the robot block, skipping all object-block compute."""
        c = self.cfg
        state = self.surgery.project_state(state, mask, lock_angle)
        action = self.surgery.project_action(action, mask)
        q, qvel, obj = state[:, :c.dof], state[:, c.dof:2*c.dof], state[:, 2*c.dof:]
        depth = torch.linspace(0.0, 1.0, c.dof, device=state.device, dtype=state.dtype)
        depth = depth.view(1, -1).expand(state.shape[0], -1)
        features = torch.stack((q, qvel, action, mask, lock_angle, depth), dim=-1)
        if self.contact_conditioned_robot:
            context_obj = obj if robot_context is None else robot_context
            if self.contact_gated_object_context:
                gate = pusher_box_contact_gate(
                    q, obj[:, :2], threshold=self.reaction_gate_threshold,
                    temperature=self.reaction_gate_temperature,
                )
                context_obj = context_obj * gate.unsqueeze(-1)
            object_context = context_obj.unsqueeze(1).expand(-1, c.dof, -1)
            features = torch.cat((features, object_context), dim=-1)
        robot_hidden = hidden
        if self.independent_object_encoder and hidden is not None:
            robot_hidden = (hidden[0] if isinstance(hidden, tuple)
                            else hidden[:, :c.dof * self.robot_expert_count])
        elif self.reaction_event_decay is not None and isinstance(hidden, tuple):
            robot_hidden = hidden[0]
        hidden_chunks = ([None] * self.robot_expert_count if robot_hidden is None
                         else list(robot_hidden.split(c.dof, dim=1)))
        experts = [(self.robot_encoder, self.robot_message, self.robot_update,
                    self.robot_temporal, self.robot_head)]
        experts.extend((item["encoder"], item["message"], item["updater"],
                        item["temporal"], item["head"])
                       for item in self.additional_robot_experts)
        next_hiddens, deltas = [], []
        for index, (encoder, message_net, update, temporal, head) in enumerate(experts):
            nodes = encoder(features)
            for _ in range(c.message_steps):
                messages = message_net(torch.cat((nodes, self._neighbor_sum(nodes)), -1))
                nodes = update(messages.flatten(0, 1), nodes.flatten(0, 1)).view_as(nodes)
            prior = hidden_chunks[index]
            if prior is None:
                prior = torch.zeros_like(nodes)
            expert_hidden = temporal(
                nodes.flatten(0, 1), prior.flatten(0, 1)
            ).view_as(nodes)
            next_hiddens.append(expert_hidden)
            deltas.append(head(expert_hidden))
        next_hidden = torch.cat(next_hiddens, dim=1)
        delta = torch.stack(deltas).mean(0)
        next_qvel = qvel + self.robot_velocity_delta_scale * delta[..., 1]
        next_q = q + self.robot_position_delta_scale * delta[..., 0]
        if self.kinematic_integration_dt is not None:
            integrated_q = q + self.kinematic_integration_dt * next_qvel
            blend = self.kinematic_position_blend
            next_q = (1.0 - blend) * next_q + blend * integrated_q
        robot = torch.cat((next_q, next_qvel), -1)
        provisional = torch.cat((robot, obj), -1)
        projected_robot = self.surgery.project_state(provisional, mask, lock_angle)[:, :2*c.dof]
        if self.reaction_rank > 0:
            context = obj.unsqueeze(1).expand(-1, c.dof, -1)
            reaction_input = features if self.reaction_physical_features else torch.cat(
                (next_hidden, context), -1
            )
            reaction = self.reaction_adapter(reaction_input)
            reaction = reaction * self.reaction_scale
            if self.reaction_geometry_gate:
                contact_gate = self._reaction_contact_gate(q, obj[:, :2])
                if self.reaction_event_decay is not None:
                    previous_trace = (hidden[2] if isinstance(hidden, tuple) and len(hidden) == 3
                                      else state.new_zeros(state.shape[0]))
                    contact_gate = torch.maximum(contact_gate,
                                                 self.reaction_event_decay * previous_trace)
                    reaction_trace = contact_gate
                reaction = reaction * contact_gate.view(-1, 1, 1)
            reaction = reaction * (1.0 - mask).unsqueeze(-1)
            if self.reaction_relative_clip is not None:
                base_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                reaction_norm = torch.linalg.vector_norm(reaction, dim=-1, keepdim=True)
                limit = self.reaction_relative_clip * base_norm
                reaction = reaction * torch.clamp(
                    limit / reaction_norm.clamp_min(1e-12), max=1.0
                )
            corrected = projected_robot.clone()
            corrected[:, :c.dof] += reaction[..., 0]
            corrected[:, c.dof:] += reaction[..., 1]
            projected_robot = self.surgery.project_state(
                torch.cat((corrected, obj), -1), mask, lock_angle)[:, :2*c.dof]
        if self.linear_physical_reaction:
            reaction = self.linear_reaction_adapter(features)
            reaction = reaction * (1.0 - mask).unsqueeze(-1)
            corrected = projected_robot.clone()
            corrected[:, :c.dof] += reaction[..., 0]
            corrected[:, c.dof:] += reaction[..., 1]
            projected_robot = self.surgery.project_state(
                torch.cat((corrected, obj), -1), mask, lock_angle
            )[:, :2*c.dof]
        if self.reaction_event_decay is not None:
            if not self.reaction_geometry_gate:
                raise RuntimeError("event decay requires reaction_geometry_gate")
            next_hidden = (next_hidden, reaction_trace)
        return projected_robot, next_hidden, obj, action, depth

    def step_object(
        self, projected_robot, obj, action, mask, lock_angle, depth, robot_hidden,
        object_hidden=None, previous_robot=None,
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
            bridge = robot_hidden.mean(1).detach()
            if not self.compact_bridge_object_head:
                bridge = torch.cat((bridge, projected_robot.detach()), -1)
            next_obj = obj + self.object_head(torch.cat((bridge, obj), -1))
            returned_hidden = robot_hidden
        if self.geometric_object_rank > 0:
            if previous_robot is None:
                raise ValueError("geometric object propagation requires previous_robot")
            previous_q = previous_robot[:, :c.dof]
            projected_q = projected_robot[:, :c.dof]
            previous_tip = pusher_reference_point(previous_q)[..., :2]
            projected_tip = pusher_reference_point(projected_q)[..., :2]
            previous_gate = pusher_box_contact_gate(
                previous_q, obj[:, :2], threshold=self.reaction_gate_threshold,
                temperature=self.reaction_gate_temperature)
            projected_gate = pusher_box_contact_gate(
                projected_q, obj[:, :2], threshold=self.reaction_gate_threshold,
                temperature=self.reaction_gate_temperature)
            geometry = torch.cat((
                previous_tip, projected_tip, projected_tip - previous_tip,
                projected_tip - obj[:, :2], obj[:, 2:4],
                previous_gate[:, None], projected_gate[:, None],
            ), -1).detach()
            next_obj = next_obj + self.geometric_object_head(geometry)
        prediction = torch.cat((projected_robot, next_obj), -1)
        return self.surgery.project_state(prediction, mask, lock_angle), returned_hidden
