from __future__ import annotations

import torch
from torch import nn

from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout


class ToyModel(nn.Module):
    def __init__(self, robot_delta: float, object_delta: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.robot_delta = robot_delta
        self.object_delta = object_delta
        self.context = None

    def set_residual_context(self, context):
        self.context = context

    def step(self, state, action, mask, lock_angle, hidden=None):
        result = state.clone()
        result[:, :10] += self.robot_delta
        result[:, 10:] += self.object_delta
        return result, hidden


class RobotCoupledToyModel(ToyModel):
    def step(self, state, action, mask, lock_angle, hidden=None):
        result = state.clone()
        result[:, :10] += self.robot_delta
        result[:, 10:] += self.object_delta + state[:, :1]
        return result, hidden


def test_selective_rollout_publishes_carrier_robot_and_intervention_object():
    full = ToyModel(robot_delta=9.0, object_delta=2.0)
    carrier = ToyModel(robot_delta=1.0, object_delta=0.5)
    model = SelectiveInterventionRollout(full, carrier)
    state = torch.zeros(1, 14)
    action = torch.zeros(1, 5)
    mask = torch.zeros(1, 5)
    angle = torch.zeros(1, 5)

    first, hidden = model.step(state, action, mask, angle)
    second, _ = model.step(first, action, mask, angle, hidden)

    assert torch.allclose(first[:, :10], torch.ones(1, 10))
    assert torch.allclose(first[:, 10:], torch.full((1, 4), 2.0))
    assert torch.allclose(second[:, :10], torch.full((1, 10), 2.0))
    assert torch.allclose(second[:, 10:], torch.full((1, 4), 4.0))


def test_selective_rollout_preserves_locked_coordinates():
    model = SelectiveInterventionRollout(
        ToyModel(robot_delta=9.0, object_delta=2.0),
        ToyModel(robot_delta=1.0, object_delta=0.5),
    )
    state = torch.zeros(1, 14)
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    angle = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0]])
    output, _ = model.step(state, torch.zeros(1, 5), mask, angle)
    assert float(output[0, 2]) == torch.tensor(0.3).item()
    assert float(output[0, 7]) == 0.0


def test_intervention_keeps_its_internal_coupled_state():
    model = SelectiveInterventionRollout(
        RobotCoupledToyModel(robot_delta=9.0, object_delta=2.0),
        ToyModel(robot_delta=1.0, object_delta=0.5),
    )
    state = torch.zeros(1, 14)
    action = torch.zeros(1, 5)
    mask = torch.zeros(1, 5)
    angle = torch.zeros(1, 5)

    first, hidden = model.step(state, action, mask, angle)
    second, _ = model.step(first, action, mask, angle, hidden)

    assert torch.allclose(second[:, :10], torch.full((1, 10), 2.0))
    assert torch.allclose(second[:, 10:], torch.full((1, 4), 13.0))


def test_projection_and_robot_isolation_hold_for_random_25_step_batches():
    torch.manual_seed(20260829)
    intervention = ToyModel(robot_delta=1_000.0, object_delta=0.7)
    carrier = ToyModel(robot_delta=-0.125, object_delta=0.0)
    model = SelectiveInterventionRollout(intervention, carrier)
    state = torch.randn(8, 14)
    mask = torch.zeros(8, 5); mask[torch.arange(8), torch.arange(8) % 5] = 1.0
    angle = torch.randn(8, 5) * mask
    action = torch.randn(8, 5)
    expected_carrier = state.clone(); hidden = None
    for _ in range(25):
        expected_carrier[:, :10] -= 0.125
        expected_carrier[:, :5] = expected_carrier[:, :5] * (1.0 - mask) + angle * mask
        expected_carrier[:, 5:10] *= 1.0 - mask
        state, hidden = model.step(state, action, mask, angle, hidden)
        torch.testing.assert_close(state[:, :10], expected_carrier[:, :10], rtol=0, atol=0)
        assert torch.count_nonzero((state[:, :5] - angle) * mask) == 0
        assert torch.count_nonzero(state[:, 5:10] * mask) == 0
