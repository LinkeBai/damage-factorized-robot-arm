"""Non-learning Reach controllers used to validate and excite the simulator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robotarm.envs.fk import forward_kinematics, inverse_kinematics


@dataclass(frozen=True)
class JacobianReachConfig:
    kp: float = 50.0
    kd: float = 1.0
    finite_difference: float = 1e-4


@dataclass(frozen=True)
class JointReferenceConfig:
    kp: float = 5.0
    kd: float = 0.5
    global_samples: int = 20_000
    seed: int = 7


def directional_push_waypoints(
    block_xy: npt.ArrayLike,
    target_xy: npt.ArrayLike,
    *,
    pusher_offset_m: float = 0.03,
    height_m: float = 0.025,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Construct direction-aware pre-contact and terminal pusher waypoints.

    The pusher starts behind the block relative to the desired motion and ends
    behind the target by the same tool-to-block offset.  Unlike the historical
    fixed +x offset, this definition is valid for arbitrary planar goals.
    """
    block = np.asarray(block_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if block.shape != (2,) or target.shape != (2,):
        raise ValueError("block_xy and target_xy must both have shape (2,)")
    delta = target - block
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        raise ValueError("push target must differ from the block position")
    direction = delta / distance
    approach_xy = block - float(pusher_offset_m) * direction
    terminal_xy = target - float(pusher_offset_m) * direction
    return (
        np.array([*approach_xy, float(height_m)], dtype=np.float64),
        np.array([*terminal_xy, float(height_m)], dtype=np.float64),
    )


def position_jacobian(
    q: npt.ArrayLike, *, epsilon: float = 1e-4
) -> npt.NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    origin = forward_kinematics(q)
    jacobian = np.zeros((3, 5), dtype=np.float64)
    for joint in range(5):
        probe = q.copy()
        probe[joint] += epsilon
        jacobian[:, joint] = (forward_kinematics(probe) - origin) / epsilon
    return jacobian


def jacobian_reach_action(
    state: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    locked_joints: tuple[int, ...] = (),
    config: JacobianReachConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Return bounded torque action from Cartesian error and joint damping."""
    cfg = config or JacobianReachConfig()
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError(f"state must have shape (10,), got {state.shape}")
    q, qvel = state[:5], state[5:]
    error = np.asarray(target, dtype=np.float64) - forward_kinematics(q)
    action = cfg.kp * position_jacobian(
        q, epsilon=cfg.finite_difference
    ).T @ error - cfg.kd * qvel
    action = np.clip(action, -1.0, 1.0)
    if locked_joints:
        action[list(locked_joints)] = 0.0
    return action


def solve_reach_reference(
    target: npt.ArrayLike,
    joint_ranges: npt.ArrayLike,
    *,
    locked_joints: dict[int, float] | None = None,
    config: JointReferenceConfig | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    """Solve position IK from a deterministic global-sampling initialization."""
    cfg = config or JointReferenceConfig()
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)
    candidates = rng.uniform(ranges[:, 0], ranges[:, 1], (cfg.global_samples, 5))
    for joint, angle in (locked_joints or {}).items():
        candidates[:, joint] = angle
    positions = np.stack([forward_kinematics(q) for q in candidates])
    q0 = candidates[np.argmin(np.linalg.norm(positions - target, axis=1))]
    return inverse_kinematics(
        target,
        ranges,
        q0=q0,
        locked=locked_joints,
        max_steps=1_000,
    )


def joint_reference_action(
    state: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    locked_joints: tuple[int, ...] = (),
    config: JointReferenceConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Track a globally solved joint reference with bounded damped PD torque."""
    cfg = config or JointReferenceConfig()
    state = np.asarray(state, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    action = np.clip(
        cfg.kp * (reference - state[:5]) - cfg.kd * state[5:], -1.0, 1.0
    )
    if locked_joints:
        action[list(locked_joints)] = 0.0
    return action
