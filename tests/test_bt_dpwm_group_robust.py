import pytest
import torch

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphWorldModel
from scripts.run_bt_dpwm_gate_y0 import (
    aggregate_topology_losses, object_teacher_losses_per_trajectory,
)


def test_group_robust_objective_interpolates_mean_and_worst_topology():
    losses = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True)
    groups = {
        "D1": torch.tensor([0, 1]),
        "D4": torch.tensor([2, 3]),
    }

    average, values = aggregate_topology_losses(losses, groups, 0.0)
    robust, _ = aggregate_topology_losses(losses, groups, 0.5)
    worst, _ = aggregate_topology_losses(losses, groups, 1.0)

    assert values["D1"].item() == pytest.approx(2.0)
    assert values["D4"].item() == pytest.approx(3.0)
    assert average.item() == pytest.approx(2.5)
    assert robust.item() == pytest.approx(2.75)
    assert worst.item() == pytest.approx(3.0)
    robust.backward()
    assert losses.grad is not None


def test_group_robust_weight_is_bounded():
    losses = torch.ones(2)
    groups = {"D1": torch.tensor([0]), "D2": torch.tensor([1])}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_topology_losses(losses, groups, 1.1)


def test_object_teacher_loss_updates_student_but_not_frozen_teacher():
    student, teacher = BlockTriangularDPWM(), TopologyGraphWorldModel()
    states = torch.randn(2, 4, 14)
    actions = torch.randn(2, 3, 5)
    mask = torch.zeros(2, 5)
    angle = torch.zeros(2, 5)
    loss = object_teacher_losses_per_trajectory(
        student, teacher, (states, actions, mask, angle), horizon=3).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.object_head.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
