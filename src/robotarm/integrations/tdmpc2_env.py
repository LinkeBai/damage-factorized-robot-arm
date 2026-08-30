"""TD-MPC2-compatible original 5-DoF hard-lock Push environment.

The adapter deliberately keeps the original ``arm_push.xml`` as the primary
platform.  It exposes the legacy four-return API expected by the reproduced
TD-MPC2 code while preserving the exact joint-lock projection implemented by
``MujocoArmEnv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robotarm.envs.damage import DamageConfig
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import (
    directional_push_waypoints,
    joint_reference_action,
    solve_reach_reference,
)

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "TD-MPC2 integration requires gymnasium==0.29.1"
    ) from exc


class OriginalArmPushEnv(gym.Env):
    """Goal-conditioned state Push task for matched TD-MPC2 experiments.

    Observation layout (33 floats): underlying 14-D arm/block state, 3-D
    end-effector position, 2-D target, block-to-target delta, end-effector-to-
    block delta, 5-D lock mask, and 5-D lock angle.  Supplying the oracle fault
    descriptor creates a strong conditioned baseline; later ablations can hide
    it and infer status from history.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        damage: DamageConfig | None = None,
        xml_path: str | Path | None = None,
        block_initial_xy: tuple[float, float] = (0.24, 0.10),
        target_x_range: tuple[float, float] = (0.18, 0.215),
        target_y_range: tuple[float, float] = (0.09, 0.11),
        success_tolerance_m: float = 0.01,
        max_episode_steps: int = 150,
        seed_policy: str = "random",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.damage = damage.copy() if damage is not None else DamageConfig.intact()
        self._block_initial_xy = np.asarray(block_initial_xy, dtype=np.float64)
        self._target_x_range = tuple(float(v) for v in target_x_range)
        self._target_y_range = tuple(float(v) for v in target_y_range)
        self.success_tolerance_m = float(success_tolerance_m)
        self.max_episode_steps = int(max_episode_steps)
        if seed_policy not in {"random", "directional", "directional_unaware"}:
            raise ValueError(
                "seed_policy must be 'random', 'directional', or "
                "'directional_unaware'"
            )
        self.seed_policy = seed_policy
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[3] / "sim/assets/arm_push.xml"
        self._env = MujocoArmEnv(xml_path=xml_path, block_initial_xy=self._block_initial_xy)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(33,), dtype=np.float32
        )
        self._target_xy = np.zeros(2, dtype=np.float64)
        self._initial_block_xy = self._block_initial_xy.copy()
        self._previous_goal_distance = 0.0
        self._step = 0
        self._seed_references: tuple[np.ndarray, np.ndarray] | None = None
        self._unaware_seed_references: tuple[np.ndarray, np.ndarray] | None = None

    def _sample_target(self) -> np.ndarray:
        # Exclude near-trivial goals: every target starts at least 25 mm away.
        for _ in range(1_000):
            target = np.array(
                [
                    self._rng.uniform(*self._target_x_range),
                    self._rng.uniform(*self._target_y_range),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(target - self._block_initial_xy) >= 0.025:
                return target
        raise RuntimeError("target ranges do not contain a non-trivial Push goal")

    def _observation(self) -> np.ndarray:
        state = self._env._observe()["state"]
        ee = self._env.ee_pos()
        block = self._env.block_pos()
        return np.concatenate(
            [
                state,
                ee,
                self._target_xy,
                self._target_xy - block,
                block - ee[:2],
                self.damage.joint_mask.astype(np.float64),
                self.damage.lock_angle,
            ]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        self._target_xy = self._sample_target()
        target_xyz = np.array([*self._target_xy, 0.025], dtype=np.float64)
        self._env.reset(target=target_xyz, damage_config=self.damage)
        self._initial_block_xy = self._env.block_pos().copy()
        self._previous_goal_distance = float(
            np.linalg.norm(self._env.block_pos() - self._target_xy)
        )
        self._step = 0
        self._seed_references = None
        self._unaware_seed_references = None
        if self.seed_policy in {"directional", "directional_unaware"}:
            approach, terminal = directional_push_waypoints(
                self._initial_block_xy, self._target_xy
            )
            locked = {
                joint: self.damage.lock_angle_of(joint)
                for joint in self.damage.locked
            }
            self._seed_references = (
                solve_reach_reference(
                    approach, self._env.joint_ranges, locked_joints=locked
                )[0],
                solve_reach_reference(
                    terminal, self._env.joint_ranges, locked_joints=locked
                )[0],
            )
            self._unaware_seed_references = (
                solve_reach_reference(approach, self._env.joint_ranges)[0],
                solve_reach_reference(terminal, self._env.joint_ranges)[0],
            )
        # Reproduced TD-MPC2 expects reset() -> observation, not Gymnasium tuple.
        return self._observation()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._env.step(action)
        self._step += 1
        block = self._env.block_pos()
        ee = self._env.ee_pos()
        goal_distance = float(np.linalg.norm(block - self._target_xy))
        approach_distance = float(np.linalg.norm(ee[:2] - block))
        progress = self._previous_goal_distance - goal_distance
        success = goal_distance <= self.success_tolerance_m
        reward = (
            20.0 * progress
            - 2.0 * goal_distance
            - 0.2 * approach_distance
            - 0.01 * float(np.square(action).sum())
            + 10.0 * float(success)
        )
        self._previous_goal_distance = goal_distance
        timeout = self._step >= self.max_episode_steps
        done = bool(success or timeout)
        info = {
            "success": bool(success),
            "terminated": bool(success),
            "timeout": bool(timeout and not success),
            "goal_distance_m": goal_distance,
            "block_xy": block.astype(np.float32),
            "target_xy": self._target_xy.astype(np.float32),
            "contact": bool(
                self._env.last_has_contact("tool_geom", "block_geom")
                or self._env.last_has_contact("pusher_geom", "block_geom")
            ),
            "block_displacement_m": float(
                np.linalg.norm(block - self._initial_block_xy)
            ),
        }
        return self._observation(), float(reward), done, info

    def rand_act(self):
        if self.seed_policy == "random":
            return self._rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        action = self.directional_action(
            fault_aware=self.seed_policy == "directional"
        ).astype(np.float64)
        action += self._rng.normal(0.0, 0.03, size=5)
        action = np.clip(action, -1.0, 1.0)
        action[list(self.damage.locked)] = 0.0
        return action.astype(np.float32)

    def directional_action(self, *, fault_aware: bool) -> np.ndarray:
        """Return deterministic directional control at the current state.

        ``fault_aware=False`` preserves the intact IK reference and only zeros
        the failed coordinate. ``True`` recomputes the reference under the hard
        lock and therefore serves as the constrained oracle label.
        """
        references = (
            self._seed_references if fault_aware else self._unaware_seed_references
        )
        if references is None:
            raise RuntimeError(
                "directional_action requires a directional seed_policy"
            )
        phase = 0 if self._step < 0.4 * self.max_episode_steps else 1
        state = self._env._observe()["state"][:10]
        action = joint_reference_action(
            state,
            references[phase],
            locked_joints=tuple(self.damage.locked),
        )
        action[list(self.damage.locked)] = 0.0
        return action.astype(np.float32)

    def close(self) -> None:
        self._env.close()
