import numpy as np
import pytest

from robotarm.deployment.vision_pose import (
    object_in_reference, planar_pose, validate_camera_calibration, VelocityFilter)


def test_camera_calibration_validation():
    matrix, distortion, size = validate_camera_calibration(
        [[500, 0, 320], [0, 510, 240], [0, 0, 1]], [0, 0, 0, 0, 0], .04)
    assert matrix.shape == (3, 3) and distortion.shape == (5,) and size == .04
    with pytest.raises(ValueError):
        validate_camera_calibration(np.eye(2), [0]*5, .04)
    with pytest.raises(ValueError):
        validate_camera_calibration(np.eye(3), [0]*3, .04)


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
