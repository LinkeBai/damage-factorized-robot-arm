import numpy as np
import pytest

from robotarm.deployment.real_calibration import (
    CalibrationPlan, build_state, radians_to_ticks, safe_excitation,
    ticks_to_radians, validate_transition_arrays)


def test_tick_conversion_round_trip():
    zero = np.array([2023, 2066, 2058, 2076, 2066])
    direction = np.ones(5)
    angle = np.array([-.2, -.1, 0., .1, .2])
    restored = ticks_to_radians(radians_to_ticks(angle, zero, direction), zero, direction)
    assert np.allclose(restored, angle, atol=2*np.pi/4096)


def test_safe_excitation_is_bounded_and_projects_lock():
    plan = CalibrationPlan("D3", 2, 50, 10.0, 0.04, 7)
    first, second = safe_excitation(plan), safe_excitation(plan)
    assert np.array_equal(first, second)
    assert first.shape == (50, 5)
    assert np.max(np.abs(first)) <= .04
    assert np.array_equal(first[:, 2], np.zeros(50))


def test_transition_validation_rejects_lock_drift():
    states = np.zeros((6, 14)); actions = np.zeros((5, 5))
    states[-1, 1] = .2
    with pytest.raises(ValueError, match="drift"):
        validate_transition_arrays(states, actions, 1, 0., .05)


def test_build_state_requires_finite_camera_pose():
    with pytest.raises(ValueError, match="14 finite"):
        build_state(np.zeros(5), np.zeros(5), [np.nan, 0, 0, 0])
