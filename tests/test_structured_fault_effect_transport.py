import numpy as np
import pytest

from robotarm.models.structured_fault_effect_transport import (
    SFETConfig,
    StructuredFaultEffectTransport,
)


def test_repair_enforces_locked_action_exactly():
    model = StructuredFaultEffectTransport(
        [[1.0, 0.5, 0.2], [0.0, 1.0, -0.5]], locked=(1,)
    )
    repaired = model.repair([0.3, 0.8, -0.2], [0.4, -0.1], [0.0, 0.0])
    assert repaired[1] == pytest.approx(0.0, abs=0.0)
    assert np.all(np.abs(repaired) <= 1.0)


def test_broyden_update_satisfies_observed_free_joint_secant():
    model = StructuredFaultEffectTransport(
        np.zeros((2, 3)), locked=(1,), config=SFETConfig(secant_epsilon=1e-12)
    )
    action_delta = np.array([0.4, 0.9, -0.2])
    effect_delta = np.array([0.3, -0.1])
    before = np.linalg.norm(effect_delta - model.predicted_effect_change(action_delta))
    model.update(action_delta, effect_delta)
    after = np.linalg.norm(effect_delta - model.predicted_effect_change(action_delta))
    assert after < 1e-10
    assert after < before
    np.testing.assert_array_equal(model.jacobian[:, 1], 0.0)


def test_repair_reduces_linearized_task_effect_error():
    jacobian = np.array([[1.0, 0.0, 0.4], [0.2, 0.0, 1.0]])
    model = StructuredFaultEffectTransport(
        jacobian, locked=(1,), config=SFETConfig(ridge=1e-6, action_limit=2.0)
    )
    nominal = np.array([0.1, 0.7, 0.1])
    desired = np.array([0.6, -0.3])
    baseline_effect = jacobian @ (nominal * model.free_mask)
    repaired = model.repair(nominal, desired, baseline_effect)
    repaired_effect = baseline_effect + jacobian @ (
        repaired - nominal * model.free_mask
    )
    baseline_error = np.linalg.norm(baseline_effect - desired)
    repaired_error = np.linalg.norm(repaired_effect - desired)
    assert repaired_error < 1e-4
    assert repaired_error < baseline_error
    assert repaired[1] == 0.0
