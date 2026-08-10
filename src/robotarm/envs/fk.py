"""Nominal analytic kinematics for the five-joint GenkiArm chain.

The transform order matches ``sim/assets/arm.xml``. The gripper-open servo is
not part of the positioning chain.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

N_JOINTS = 5
BASE_HEIGHT = 0.120
SHOULDER_X = 0.0
L_UPPER = 0.110
L_FOREARM = 0.120
WRIST_OFFSET = 0.060
J5_TO_J6_Y = -0.0132
J5_TO_J6_Z = 0.110
TCP_OFFSET_X = 0.020  # provisional; replace with measured gripper TCP
L_TOOL = J5_TO_J6_Z
L_DISTAL = L_FOREARM + WRIST_OFFSET + J5_TO_J6_Z


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def forward_pose(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the 4x4 world transform of the provisional gripper TCP."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"q must have shape ({N_JOINTS},), got {q.shape}")
    return (
        _translation(0, 0, BASE_HEIGHT)
        @ _rotation_z(float(q[0]))
        @ _translation(SHOULDER_X, 0, 0)
        @ _rotation_y(float(q[1]))
        @ _translation(0, 0, L_UPPER)
        @ _rotation_y(float(q[2]))
        @ _translation(0, 0, L_FOREARM)
        @ _rotation_y(float(q[3]))
        @ _translation(0, 0, WRIST_OFFSET)
        @ _rotation_z(float(q[4]))
        @ _translation(0, J5_TO_J6_Y, J5_TO_J6_Z)
        @ _translation(TCP_OFFSET_X, 0, 0)
    )


def forward_kinematics(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the 3-D world position of the gripper TCP."""
    return forward_pose(q)[:3, 3].copy()


def inverse_kinematics(
    target: npt.ArrayLike,
    joint_ranges: npt.ArrayLike,
    *,
    q0: npt.ArrayLike | None = None,
    locked: dict[int, float] | None = None,
    max_steps: int = 250,
    tolerance: float = 1e-3,
    damping: float = 1e-3,
) -> tuple[npt.NDArray[np.float64], float]:
    """Solve position-only IK while holding diagnosed joints exactly fixed."""
    target = np.asarray(target, dtype=np.float64).reshape(3)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if ranges.shape != (N_JOINTS, 2):
        raise ValueError(
            f"joint_ranges must have shape ({N_JOINTS}, 2), got {ranges.shape}"
        )
    q = (
        np.asarray(q0, dtype=np.float64).copy()
        if q0 is not None
        else ranges.mean(axis=1)
    )
    if q.shape != (N_JOINTS,):
        raise ValueError(f"q0 must have shape ({N_JOINTS},), got {q.shape}")
    locked = locked or {}
    free = [joint for joint in range(N_JOINTS) if joint not in locked]
    for joint, angle in locked.items():
        if joint not in range(N_JOINTS):
            raise ValueError(f"locked joint index {joint} out of range")
        q[joint] = angle

    epsilon = 1e-5
    for _ in range(max_steps):
        position = forward_kinematics(q)
        error = target - position
        if np.linalg.norm(error) <= tolerance or not free:
            break
        jacobian = np.zeros((3, len(free)), dtype=np.float64)
        for column, joint in enumerate(free):
            probe = q.copy()
            probe[joint] += epsilon
            jacobian[:, column] = (
                forward_kinematics(probe) - position
            ) / epsilon
        lhs = jacobian @ jacobian.T + damping * np.eye(3)
        delta = jacobian.T @ np.linalg.solve(lhs, error)
        q[free] += np.clip(delta, -0.15, 0.15)
        q = np.clip(q, ranges[:, 0], ranges[:, 1])
        for joint, angle in locked.items():
            q[joint] = angle

    return q, float(np.linalg.norm(target - forward_kinematics(q)))
