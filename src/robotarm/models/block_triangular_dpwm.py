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
    ) -> None:
        super().__init__()
        self.cfg = cfg or TopologyGraphConfig()
        self.contact_conditioned_robot = contact_conditioned_robot
        self.independent_object_encoder = independent_object_encoder
        c = self.cfg
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
            # projected q/qvel, action, locked, lock angle, depth, object state
            object_input_dim = 6 + c.object_dim
            self.object_encoder = nn.Sequential(
                nn.Linear(object_input_dim, c.hidden_dim), nn.SiLU(),
                nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU(),
            )
            self.object_message = nn.Sequential(
                nn.Linear(2 * c.hidden_dim, c.hidden_dim), nn.SiLU(),
                nn.Linear(c.hidden_dim, c.hidden_dim),
            )
            self.object_update = nn.GRUCell(c.hidden_dim, c.hidden_dim)
            self.object_temporal = nn.GRUCell(c.hidden_dim, c.hidden_dim)
            self.object_head = nn.Sequential(
                nn.Linear(c.hidden_dim + c.object_dim, c.hidden_dim), nn.SiLU(),
                nn.Linear(c.hidden_dim, c.object_dim),
            )
        self.surgery = TopologySurgery()

    @staticmethod
    def _neighbor_sum(nodes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(nodes)
        result[:, 1:] += nodes[:, :-1]
        result[:, :-1] += nodes[:, 1:]
        return result

    def step(self, state, action, mask, lock_angle, hidden):
        object_hidden = None
        if self.independent_object_encoder and hidden is not None:
            object_hidden = hidden[:, self.cfg.dof:]
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
            robot_hidden = hidden[:, :c.dof]
        if robot_hidden is None:
            robot_hidden = torch.zeros_like(nodes)
        next_hidden = self.robot_temporal(
            nodes.flatten(0, 1), robot_hidden.flatten(0, 1)
        ).view_as(nodes)
        delta = self.robot_head(next_hidden)
        robot = torch.cat((q + delta[..., 0], qvel + delta[..., 1]), -1)
        provisional = torch.cat((robot, obj), -1)
        projected_robot = self.surgery.project_state(provisional, mask, lock_angle)[:, :2*c.dof]
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
            returned_hidden = torch.cat((robot_hidden, next_object_hidden), 1)
        else:
            bridge = torch.cat((robot_hidden.mean(1), projected_robot), -1).detach()
            next_obj = obj + self.object_head(torch.cat((bridge, obj), -1))
            returned_hidden = robot_hidden
        prediction = torch.cat((projected_robot, next_obj), -1)
        return self.surgery.project_state(prediction, mask, lock_angle), returned_hidden
