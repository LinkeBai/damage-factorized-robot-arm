import numpy as np
import pytest

from robotarm.deployment.vision_pose import object_in_reference, planar_pose, VelocityFilter


def test_relative_marker_translation_identity():
    actual = object_in_reference(np.eye(3), [1, 2, 3], [1.2, 1.7, 3.1])
    np.testing.assert_allclose(actual, [.2, -.3, .1])


def test_planar_axis_mapping():
    np.testing.assert_allclose(planar_pose([1, 2, 3], (2, 0), (-1, 1)), [-3, 1])
    with pytest.raises(ValueError):
        planar_pose([1, 2, 3], (0, 0))


def test_velocity_filter_and_gap_reset():
    tracker = VelocityFilter(smoothing=0.0, maximum_dt_s=.5)
    np.testing.assert_allclose(tracker.update([0, 0], 1.0), [0, 0])
    np.testing.assert_allclose(tracker.update([.1, -.2], 1.1), [1, -2])
    np.testing.assert_allclose(tracker.update([.2, -.3], 2.0), [0, 0])
