"""Pure geometry and filtering helpers for eye-in-hand marker tracking."""
from __future__ import annotations

import numpy as np


def object_in_reference(r_reference_camera, t_reference_camera,
                        t_object_camera) -> np.ndarray:
    """Return object-marker origin in the reference-marker coordinate frame."""
    rotation = np.asarray(r_reference_camera, dtype=float).reshape(3, 3)
    reference = np.asarray(t_reference_camera, dtype=float).reshape(3)
    obj = np.asarray(t_object_camera, dtype=float).reshape(3)
    result = rotation.T @ (obj-reference)
    if not np.isfinite(result).all():
        raise ValueError("marker transforms must be finite")
    return result


def planar_pose(relative_xyz, axes=(0, 1), signs=(1.0, 1.0)) -> np.ndarray:
    xyz = np.asarray(relative_xyz, dtype=float).reshape(3)
    if len(set(axes)) != 2 or any(index not in (0, 1, 2) for index in axes):
        raise ValueError("planar axes must be two distinct values from 0, 1, 2")
    result = xyz[np.asarray(axes)]*np.asarray(signs, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("invalid planar pose")
    return result


class VelocityFilter:
    def __init__(self, smoothing: float = 0.65, maximum_dt_s: float = 0.5):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing, self.maximum_dt_s = smoothing, maximum_dt_s
        self.previous_position = self.previous_time = None
        self.velocity = np.zeros(2, dtype=float)

    def update(self, position, timestamp: float) -> np.ndarray:
        position = np.asarray(position, dtype=float).reshape(2)
        if self.previous_time is not None:
            dt = timestamp-self.previous_time
            if 0.0 < dt <= self.maximum_dt_s:
                raw = (position-self.previous_position)/dt
                self.velocity = self.smoothing*self.velocity+(1-self.smoothing)*raw
            else:
                self.velocity[:] = 0.0
        self.previous_position, self.previous_time = position.copy(), float(timestamp)
        return self.velocity.copy()
