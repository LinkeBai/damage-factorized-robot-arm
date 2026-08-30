import numpy as np
import pytest

from robotarm.training.controllers import directional_push_waypoints


def test_leftward_push_starts_on_right_and_finishes_behind_target():
    approach, terminal = directional_push_waypoints([0.24, 0.10], [0.19, 0.10])
    np.testing.assert_allclose(approach, [0.27, 0.10, 0.025])
    np.testing.assert_allclose(terminal, [0.22, 0.10, 0.025])


def test_waypoints_follow_arbitrary_planar_direction():
    block = np.array([0.24, 0.10])
    target = np.array([0.20, 0.13])
    approach, terminal = directional_push_waypoints(block, target)
    direction = (target - block) / np.linalg.norm(target - block)
    np.testing.assert_allclose(approach[:2], block - 0.03 * direction)
    np.testing.assert_allclose(terminal[:2], target - 0.03 * direction)


def test_coincident_target_is_rejected():
    with pytest.raises(ValueError, match="must differ"):
        directional_push_waypoints([0.24, 0.10], [0.24, 0.10])
