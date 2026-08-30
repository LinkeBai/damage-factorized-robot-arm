"""Carrier-state rollout that isolates object intervention from robot feedback."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .topology_surgery import TopologySurgery


@dataclass
class SelectiveRolloutHidden:
    intervention_hidden: object
    carrier_hidden: object
    intervention_state: torch.Tensor
    carrier_state: torch.Tensor


class SelectiveInterventionRollout(nn.Module):
    """Keep robot/free-state rollout on the fallback path.

    The intervention model predicts only the object block.  A parallel
    no-intervention carrier advances robot state and its own object context;
    the published state combines carrier robot coordinates with intervention
    object coordinates.  This prevents an object-specific correction from
    feeding back into free-joint prediction while preserving analytic lock
    projection.
    """

    def __init__(self, intervention_model: nn.Module, carrier_model: nn.Module,
                 *, robot_dim: int = 10) -> None:
        super().__init__()
        self.intervention_model = intervention_model
        self.carrier_model = carrier_model
        self.robot_dim = int(robot_dim)
        self.surgery = TopologySurgery()

    def set_residual_context(self, context: torch.Tensor | None) -> None:
        self.intervention_model.set_residual_context(context)
        self.carrier_model.set_residual_context(context)

    def step(self, state, action, mask, lock_angle, hidden=None):
        if isinstance(hidden, SelectiveRolloutHidden):
            intervention_hidden = hidden.intervention_hidden
            carrier_hidden = hidden.carrier_hidden
            intervention_state = hidden.intervention_state
            carrier_state = hidden.carrier_state
        else:
            intervention_hidden = hidden
            carrier_hidden = None
            intervention_state = state
            carrier_state = state

        carrier_raw, next_carrier_hidden = self.carrier_model.step(
            carrier_state, action, mask, lock_angle, carrier_hidden
        )
        carrier_next = self.surgery.project_state(carrier_raw, mask, lock_angle)

        intervention_raw, next_intervention_hidden = self.intervention_model.step(
            intervention_state, action, mask, lock_angle, intervention_hidden
        )
        intervention_next = self.surgery.project_state(
            intervention_raw, mask, lock_angle
        )
        combined = torch.cat((
            carrier_next[:, : self.robot_dim],
            intervention_next[:, self.robot_dim :],
        ), dim=-1)
        combined = self.surgery.project_state(combined, mask, lock_angle)
        return combined, SelectiveRolloutHidden(
            intervention_hidden=next_intervention_hidden,
            carrier_hidden=next_carrier_hidden,
            intervention_state=intervention_next,
            carrier_state=carrier_next,
        )
