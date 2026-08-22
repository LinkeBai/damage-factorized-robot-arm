import pytest
import torch

from scripts.run_bt_dpwm_gate_y0 import aggregate_topology_losses


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
