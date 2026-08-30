"""Few-shot residual innovation for damage-projected world models.

The adapter is deliberately unable to learn a static correction: its output is
bilinear in an inferred deployment context ``z`` and a transition-dependent
basis.  Consequently ``z=0`` is exactly the frozen nominal model, rather than
another trainable branch that can silently absorb the average domain shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .topology_surgery import TopologySurgery


class ProjectedResidualInnovation(nn.Module):
    """Low-rank, context-only correction on free robot coordinates.

    The object block is intentionally excluded.  Object prediction remains a
    downstream test of whether the calibrated robot rollout improves contact
    prediction, not an additional path through which support data can leak.
    """

    def __init__(self, *, dof: int = 5, latent_dim: int = 8,
                 rank: int = 8, hidden_dim: int = 64,
                 position_limit: float | None = None,
                 velocity_limit: float | None = None,
                 factorized_context: bool = False,
                 joint_factorized_basis: bool = False,
                 memory_dim: int = 0,
                 analytic_history: bool = False,
                 history_deadband: float = 0.04,
                 shared_joint_basis: bool = False,
                 project_free_coordinates: bool = True) -> None:
        super().__init__()
        self.dof = dof
        self.latent_dim = latent_dim
        self.rank = rank
        self.factorized_context = factorized_context
        self.joint_factorized_basis = joint_factorized_basis
        self.memory_dim = int(memory_dim)
        self.analytic_history = bool(analytic_history)
        self.shared_joint_basis = bool(shared_joint_basis)
        self.project_free_coordinates = bool(project_free_coordinates)
        self.history_deadband = float(history_deadband)
        if self.analytic_history and self.memory_dim:
            raise ValueError("analytic history and learned recurrent memory are exclusive")
        if self.memory_dim < 0:
            raise ValueError("memory_dim must be non-negative")
        if factorized_context and rank != latent_dim:
            raise ValueError("factorized context requires rank == latent_dim")
        if (position_limit is None) != (velocity_limit is None):
            raise ValueError("position and velocity limits must be set together")
        limits = None if position_limit is None else torch.tensor(
            [position_limit] * dof + [velocity_limit] * dof)
        self.register_buffer("correction_limits", limits, persistent=False)
        history_dim = 5 * dof if self.analytic_history else self.memory_dim
        feature_dim = 2 * dof + dof + dof + history_dim
        self.memory_cell = (nn.GRUCell(2 * dof + dof, self.memory_dim)
                            if self.memory_dim else None)
        if shared_joint_basis and joint_factorized_basis:
            raise ValueError("choose shared or separate joint basis, not both")
        if shared_joint_basis:
            joint_feature_dim = 2 * dof + dof + history_dim + dof
            self.shared_joint_transition_basis = nn.Sequential(
                nn.Linear(joint_feature_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 2 * rank))
            self.joint_transition_bases = None
            self.transition_basis = None
        elif joint_factorized_basis:
            # Experts see the same continuous transition features but never
            # the topology mask.  Damage enters only through projected action
            # and the final analytic free-coordinate projection.
            joint_feature_dim = 2 * dof + dof + history_dim
            self.joint_transition_bases = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(joint_feature_dim, hidden_dim), nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * rank))
                for _ in range(dof)])
            self.transition_basis = None
        else:
            self.shared_joint_transition_basis = None
            self.joint_transition_bases = None
            self.transition_basis = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 2 * dof * rank),
            )
        # No bias is essential: it makes coefficients(0) exactly zero.
        self.context_coefficients = nn.Linear(latent_dim, rank, bias=False)
        if factorized_context:
            with torch.no_grad():
                self.context_coefficients.weight.copy_(torch.eye(latent_dim))
            self.context_coefficients.weight.requires_grad_(False)

    def step(self, state: torch.Tensor, action: torch.Tensor,
             damage_mask: torch.Tensor, z: torch.Tensor,
             memory: torch.Tensor | None = None):
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(state.shape[0], -1)
        if z.shape != (state.shape[0], self.latent_dim):
            raise ValueError(
                f"z must have shape ({state.shape[0]}, {self.latent_dim})"
            )
        robot = state[:, :2 * self.dof]
        projected_action = action * (1.0 - damage_mask)
        history = None
        if self.analytic_history:
            if memory is None:
                previous_action = torch.zeros_like(projected_action)
                previous_velocity = torch.zeros_like(robot[:, self.dof:])
            else:
                previous_action = memory[:, :self.dof]
                previous_velocity = memory[:, self.dof:]
            velocity = robot[:, self.dof:]
            action_delta = projected_action - previous_action
            velocity_delta = velocity - previous_velocity
            reversal = ((velocity * previous_velocity) < 0).to(state.dtype)
            deadband_crossing = (
                (projected_action.abs() >= self.history_deadband)
                != (previous_action.abs() >= self.history_deadband)).to(state.dtype)
            history = torch.cat((previous_action, action_delta, velocity_delta,
                                 reversal, deadband_crossing), dim=-1)
            memory = torch.cat((projected_action, velocity), dim=-1)
        elif self.memory_cell is not None:
            if memory is None:
                memory = state.new_zeros(state.shape[0], self.memory_dim)
            memory = self.memory_cell(
                torch.cat((robot, projected_action), dim=-1), memory)
        else:
            memory = None
        if self.shared_joint_basis:
            temporal = history if history is not None else memory
            base_parts = ((robot, projected_action) if temporal is None else
                          (robot, projected_action, temporal))
            base_features = torch.cat(base_parts, dim=-1)
            joint_id = torch.eye(self.dof, device=state.device,
                                 dtype=state.dtype)
            features = torch.cat((
                base_features[:, None, :].expand(-1, self.dof, -1),
                joint_id[None, :, :].expand(state.shape[0], -1, -1)), dim=-1)
            by_joint = self.shared_joint_transition_basis(features).view(
                state.shape[0], self.dof, 2, self.rank)
            basis = torch.cat((by_joint[:, :, 0], by_joint[:, :, 1]), dim=1)
        elif self.joint_factorized_basis:
            parts = (robot, projected_action) if history is None and memory is None else (
                robot, projected_action, history if history is not None else memory)
            features = torch.cat(parts, dim=-1)
            by_joint = torch.stack(
                [expert(features).view(state.shape[0], 2, self.rank)
                 for expert in self.joint_transition_bases], dim=1)
            basis = torch.cat((by_joint[:, :, 0], by_joint[:, :, 1]), dim=1)
        else:
            temporal = history if history is not None else memory
            parts = ((robot, projected_action, damage_mask) if temporal is None else
                     (robot, projected_action, damage_mask, temporal))
            features = torch.cat(parts, dim=-1)
            basis = self.transition_basis(features).view(
                state.shape[0], 2 * self.dof, self.rank)
        coefficients = self.context_coefficients(z)
        robot_correction = torch.einsum("bdr,br->bd", basis, coefficients)
        if self.correction_limits is not None:
            limits = self.correction_limits.to(robot_correction)
            robot_correction = limits * torch.tanh(robot_correction / limits)
        if self.project_free_coordinates:
            free = torch.cat((1.0 - damage_mask, 1.0 - damage_mask), dim=-1)
            robot_correction = robot_correction * free
        object_zeros = state.new_zeros(
            state.shape[0], state.shape[-1] - 2 * self.dof
        )
        return torch.cat((robot_correction, object_zeros), dim=-1), memory

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                damage_mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        correction, _ = self.step(state, action, damage_mask, z, None)
        return correction


@dataclass
class FewShotHidden:
    base: object
    residual: torch.Tensor | None


class FewShotProjectedModel(nn.Module):
    """Attach residual innovation to a frozen-compatible BT-DPWM interface."""

    def __init__(self, base_model: nn.Module, adapter: ProjectedResidualInnovation,
                 *, base_uses_topology: bool = True,
                 adapter_before_object: bool = True):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.base_uses_topology = base_uses_topology
        self.adapter_before_object = bool(adapter_before_object)
        self.surgery = TopologySurgery()
        self.register_buffer(
            "residual_context", torch.zeros(adapter.latent_dim), persistent=False
        )

    def set_residual_context(self, z: torch.Tensor | None) -> None:
        with torch.no_grad():
            if z is None:
                self.residual_context.zero_()
            else:
                if z.shape != self.residual_context.shape:
                    raise ValueError(
                        f"context must have shape {tuple(self.residual_context.shape)}"
                    )
                self.residual_context.copy_(z.to(self.residual_context))

    def step_with_context(self, state, action, mask, lock_angle, hidden, z):
        if isinstance(hidden, FewShotHidden):
            base_hidden, residual_hidden = hidden.base, hidden.residual
        else:
            base_hidden, residual_hidden = hidden, None
        base_mask = mask if self.base_uses_topology else torch.zeros_like(mask)
        base_angle = lock_angle if self.base_uses_topology else torch.zeros_like(lock_angle)
        correction, next_residual_hidden = self.adapter.step(
            state, action, mask, z, residual_hidden)
        # BT-DPWM exposes its triangular blocks.  Insert the calibrated robot
        # transition before the object block so object rollout is a genuine
        # downstream consequence of robot adaptation.
        if (self.adapter_before_object and hasattr(self.base_model, "step_robot")
                and hasattr(self.base_model, "step_object")):
            intervention_step = None
            if getattr(self.base_model, "intervention_context_ramp", 0.0) > 0.0:
                if base_hidden is None:
                    intervention_step = state.new_zeros(state.shape[0])
                else:
                    base_hidden, intervention_step = base_hidden
            object_hidden = None
            if getattr(self.base_model, "independent_object_encoder", False) and base_hidden is not None:
                object_hidden = (base_hidden[1] if isinstance(base_hidden, tuple)
                                 else base_hidden[:, self.adapter.dof:])
            robot, robot_hidden, obj, projected_action, depth = self.base_model.step_robot(
                state, action, base_mask, base_angle, base_hidden
            )
            robot = robot + correction[:, :2 * self.adapter.dof]
            combined = torch.cat((robot, obj), dim=-1)
            combined = self.surgery.project_state(combined, mask, lock_angle)
            prediction, next_hidden = self.base_model.step_object(
                combined[:, :2 * self.adapter.dof], obj, projected_action,
                base_mask, base_angle, depth, robot_hidden, object_hidden,
                previous_robot=state[:, :2 * self.adapter.dof],
                intervention_step=intervention_step,
            )
            if intervention_step is not None:
                next_hidden = (next_hidden, intervention_step + 1.0)
            # step_object returns its input robot block as the robot prediction.
            prediction = torch.cat((combined[:, :2 * self.adapter.dof],
                                    prediction[:, 2 * self.adapter.dof:]), dim=-1)
        else:
            prediction, next_hidden = self.base_model.step(
                state, action, base_mask, base_angle, base_hidden
            )
            prediction = prediction + correction
        prediction = self.surgery.project_state(prediction, mask, lock_angle)
        return prediction, FewShotHidden(next_hidden, next_residual_hidden)

    def step(self, state, action, mask, lock_angle, hidden):
        return self.step_with_context(
            state, action, mask, lock_angle, hidden, self.residual_context
        )
