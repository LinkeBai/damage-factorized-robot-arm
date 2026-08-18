"""MuJoCo implementation of the RobotEnv protocol (PROJECT-PLAN-V4 §4.7).

``MujocoArmEnv`` drives either the analytic ``arm.xml`` model or the visual
``genkiarm/arm_mesh.xml`` model and exposes the same
:class:`~robotarm.envs.protocol.RobotEnv` interface as the real arm. Damage is
injected by pinning locked joints rigidly at their ``lock_angle``: each step
we overwrite the locked joint's position (and zero its velocity) before any
simulation advance, so the kinematic chain behaves as if those joints were
immobilized — matching the G0 "joint locked in place" deployment described in
the plan.

Link lengths are read *off* the MJCF body offsets so FK tests and the env share
one source of truth. Only the ``ee`` site is treated as the end-effector.
"""
from __future__ import annotations

from pathlib import Path
from collections import deque

import mujoco
import numpy as np
import numpy.typing as npt

from .damage import DamageConfig
from .protocol import Observation, StepResult
from .residual_physics import ResidualPhysicsConfig

# Asset lives at the repo root next to src/: <root>/sim/assets/arm.xml.
# mujoco_env.py -> envs -> robotarm -> src -> <root>
_ROOT = Path(__file__).resolve().parents[3]
ASSET_PATH = _ROOT / "sim" / "assets" / "arm.xml"
MESH_ASSET_PATH = _ROOT / "sim" / "assets" / "genkiarm" / "arm_mesh.xml"
ASSET_PATHS = {"simple": ASSET_PATH, "mesh": MESH_ASSET_PATH}
CONTROLLED_JOINTS = ("j1", "j2", "j3", "j4", "j5")
CONTROLLED_ACTUATORS = ("m1", "m2", "m3", "m4", "m5")

# Actuation is normalized to [-1, 1]; scale to a max speed per joint (rad/s).
_DEFAULT_CTRL_SCALE = np.array([1.5, 1.8, 2.4, 1.8, 3.0])


class MujocoArmEnv:
    """A simulation-only arm environment conforming to ``RobotEnv``.

    Parameters
    ----------
    xml_path:
        Explicit path to an MJCF. Overrides ``model_variant`` when supplied.
    model_variant:
        ``"simple"`` for the kinematic/collision proxy or ``"mesh"`` for the
        full GenkiArm STL visual model. Both expose the same five-joint API.
    ctrl_scale:
        Multiplier mapping a normalized action in [-1, 1] to actuator torque /
        control input per joint. Only affects dynamics, not correctness of the
        observation/FK contract.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        ctrl_scale: npt.ArrayLike | None = None,
        *,
        model_variant: str = "simple",
        residual_physics: ResidualPhysicsConfig | None = None,
        block_initial_xy: npt.ArrayLike | None = None,
    ) -> None:
        if xml_path is None:
            try:
                xml_path = ASSET_PATHS[model_variant]
            except KeyError:
                raise ValueError(
                    f"unknown model_variant {model_variant!r}; choose from {sorted(ASSET_PATHS)}"
                ) from None
        self.xml_path = Path(xml_path)
        self.model_variant = model_variant
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._ee_site = self.model.site("ee").id
        self._goal_body = self.model.body("goal").id
        try:
            self._joint_ids = np.array(
                [self.model.joint(name).id for name in CONTROLLED_JOINTS], dtype=np.int32
            )
            self._actuator_ids = np.array(
                [self.model.actuator(name).id for name in CONTROLLED_ACTUATORS], dtype=np.int32
            )
        except KeyError as exc:
            raise ValueError(
                f"{self.xml_path} must define joints {CONTROLLED_JOINTS} and "
                f"actuators {CONTROLLED_ACTUATORS}"
            ) from exc
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._qvel_adr = self.model.jnt_dofadr[self._joint_ids]
        # Optional pushable block (Push task): detect slide joints block_x/block_y.
        self._block_qpos_adr = np.array([], dtype=np.int32)
        self._block_qvel_adr = np.array([], dtype=np.int32)
        for name in ("block_x", "block_y"):
            try:
                jid = self.model.joint(name).id
                self._block_qpos_adr = np.append(
                    self._block_qpos_adr, self.model.jnt_qposadr[jid]
                )
                self._block_qvel_adr = np.append(
                    self._block_qvel_adr, self.model.jnt_dofadr[jid]
                )
            except KeyError:
                pass
        self._residual_physics = residual_physics or ResidualPhysicsConfig()
        self._block_initial_xy = None if block_initial_xy is None else np.asarray(
            block_initial_xy, dtype=np.float64
        ).reshape(2)
        self._ctrl_scale = np.asarray(
            ctrl_scale if ctrl_scale is not None else _DEFAULT_CTRL_SCALE,
            dtype=np.float64,
        ) * self._residual_physics.actuator_scale_array
        if self._ctrl_scale.shape != (len(CONTROLLED_JOINTS),):
            raise ValueError(
                f"ctrl_scale must have length {len(CONTROLLED_JOINTS)}, "
                f"got {self._ctrl_scale.shape}"
            )
        self.model.dof_damping[self._qvel_adr] *= self._residual_physics.damping_scale
        self.model.dof_frictionloss[self._qvel_adr] *= self._residual_physics.friction_scale
        self.model.dof_armature[self._qvel_adr] *= self._residual_physics.armature_scale
        if self._residual_physics.payload_mass_delta_kg > 0:
            tool_id = self.model.body("tool").id
            old_mass = float(self.model.body_mass[tool_id])
            new_mass = old_mass + self._residual_physics.payload_mass_delta_kg
            self.model.body_mass[tool_id] = new_mass
            if old_mass > 0:
                self.model.body_inertia[tool_id] *= new_mass / old_mass
            mujoco.mj_setConst(self.model, self.data)

        self._dof = len(CONTROLLED_JOINTS)
        self._target = np.zeros(3)
        self._damage: DamageConfig | None = None
        self._t = 0
        self._max_steps = 1_000
        self._episode_index = 0
        self._rng = np.random.default_rng(self._residual_physics.seed)
        self._action_queue: deque[npt.NDArray[np.float64]] = deque()
        self._last_applied_action = np.zeros(self._dof, dtype=np.float64)
        self._backlash = self._residual_physics.backlash_array.copy()
        self._prev_qvel_sign = np.zeros(self._dof, dtype=np.int8)

    # ------------------------------------------------------------------
    # RobotEnv protocol
    # ------------------------------------------------------------------
    def reset(self, *, target: npt.NDArray[np.float64], damage_config: DamageConfig | None = None) -> Observation:
        self._damage = damage_config
        self._target = np.asarray(target, dtype=np.float64).reshape(-1)
        if self._target.shape != (3,):
            raise ValueError(f"target must be 3-D, got {self._target.shape}")

        self.model.body(self._goal_body).pos = self._target
        mujoco.mj_resetData(self.model, self.data)
        if self._block_initial_xy is not None and len(self._block_qpos_adr) == 2:
            block_origin = self.model.body("block").pos[:2]
            self.data.qpos[self._block_qpos_adr] = self._block_initial_xy - block_origin
        self._rng = np.random.default_rng(
            self._residual_physics.seed + self._episode_index
        )
        self._episode_index += 1
        self._action_queue = deque(
            np.zeros(self._dof, dtype=np.float64)
            for _ in range(self._residual_physics.control_delay_steps)
        )
        self._last_applied_action = np.zeros(self._dof, dtype=np.float64)
        self._prev_qvel_sign = np.zeros(self._dof, dtype=np.int8)

        # If a joint is locked, pin it at its lock angle even in the initial pose.
        self._apply_damage()
        mujoco.mj_forward(self.model, self.data)
        self._t = 0
        return self._observe()

    def step(self, action: npt.NDArray[np.float64]) -> StepResult:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self._dof,):
            raise ValueError(f"action must have shape ({self._dof},), got {action.shape}")
        filtered = np.where(
            np.abs(action) < self._residual_physics.action_deadband, 0.0, action
        )
        self._action_queue.append(filtered.copy())
        applied = self._action_queue.popleft()
        self._last_applied_action = applied.copy()
        # Normalized action -> ctrl (torque-style control).
        ctrl = applied * self._ctrl_scale
        ctrl_range = self.model.actuator_ctrlrange[self._actuator_ids]
        self.data.ctrl[self._actuator_ids] = np.clip(
            ctrl, ctrl_range[:, 0], ctrl_range[:, 1]
        )

        # Backlash: on velocity sign reversal, the transmission gap absorbs the
        # command and dissipates kinetic energy (loose gear / worn servo). This
        # is a history-dependent, non-linear effect that simple topology
        # conditioning cannot capture.
        if np.any(self._backlash > 0):
            qvel = self.data.qvel[self._qvel_adr]
            for i in range(self._dof):
                b = self._backlash[i]
                if b <= 0:
                    continue
                cur_sign = 1 if qvel[i] > 1e-4 else (-1 if qvel[i] < -1e-4 else 0)
                prev_sign = int(self._prev_qvel_sign[i])
                if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
                    self.data.ctrl[self._actuator_ids[i]] = 0.0
                    self.data.qvel[self._qvel_adr[i]] *= 0.5
                if cur_sign != 0:
                    self._prev_qvel_sign[i] = cur_sign

        mujoco.mj_step(self.model, self.data)
        # Re-pin locked joints after integration so damage is preserved exactly.
        self._apply_damage()
        mujoco.mj_forward(self.model, self.data)

        self._t += 1
        obs = self._observe()
        reward = self._reward(obs)
        success = self._success(obs)
        done = self._t >= self._max_steps
        return {
            "observation": obs,
            "reward": reward,
            "success": success,
            "done": done,
        }

    def emergency_stop(self) -> None:
        """Zero all controls and stop integrating; safe for sim."""
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self._action_queue.clear()
        self._last_applied_action[:] = 0.0

    def close(self) -> None:
        # MuJoCo has no explicit close; no-op keeps the interface uniform.
        self.data = mujoco.MjData(self.model)

    @property
    def action_dim(self) -> int:
        return self._dof

    @property
    def observation_dim(self) -> int:
        # state = joint positions (nq) + joint velocities (nv) + optional block.
        return 2 * self._dof + 2 * len(self._block_qpos_adr)

    # ------------------------------------------------------------------
    # Damage handling
    # ------------------------------------------------------------------
    def _apply_damage(self) -> None:
        """Freeze locked joints at their lock angles (G0 'locked' deployment)."""
        if self._damage is None or self._damage.n_locked == 0:
            return
        for i in self._damage.locked:
            if i >= self._dof:
                continue
            self.data.qpos[self._qpos_adr[i]] = self._damage.lock_angle_of(i)
            self.data.qvel[self._qvel_adr[i]] = 0.0
            self.data.ctrl[self._actuator_ids[i]] = 0.0

    @property
    def damage_config(self) -> DamageConfig | None:
        return self._damage

    # ------------------------------------------------------------------
    # Observations / task reward
    # ------------------------------------------------------------------
    def _observe(self) -> Observation:
        qpos = self.data.qpos[self._qpos_adr].copy()
        qvel = self.data.qvel[self._qvel_adr].copy()
        state = np.concatenate([qpos, qvel])
        if len(self._block_qpos_adr) > 0:
            state = np.concatenate([state, self.block_state()])
        if self._residual_physics.observation_noise_std > 0:
            state += self._rng.normal(
                0.0, self._residual_physics.observation_noise_std, state.shape
            )
        return {
            "state": state,
            "target": self._target.copy(),
        }

    def ee_pos(self) -> npt.NDArray[np.float64]:
        """Current end-effector position in world frame."""
        return self.data.site_xpos[self._ee_site].copy()

    def block_state(self) -> npt.NDArray[np.float64]:
        """Pushable-block world position and linear velocity, or empty if none."""
        if len(self._block_qpos_adr) == 0:
            return np.zeros(0, dtype=np.float64)
        pos = self.data.body("block").xpos[:2].copy()
        vel = self.data.body("block").cvel[:2].copy()
        return np.concatenate([pos, vel])

    def block_pos(self) -> npt.NDArray[np.float64]:
        """Pushable-block world position (2-D in the table plane), or empty."""
        if len(self._block_qpos_adr) == 0:
            return np.zeros(0, dtype=np.float64)
        return self.data.body("block").xpos[:2].copy()

    def has_contact(self, geom_a: str, geom_b: str) -> bool:
        """Return whether the named geom pair is in contact this step."""
        first = self.model.geom(geom_a).id
        second = self.model.geom(geom_b).id
        pair = {first, second}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self.data.contact
        )

    def _ee_target_error(self, obs: Observation | None = None) -> float:
        return float(np.linalg.norm(self.ee_pos() - self._target))

    def _success(self, obs: Observation | None = None) -> bool:
        # Engineering starting threshold; to be re-mapped after G0 (§ plan: 5 cm).
        return self._ee_target_error(obs) < 0.05

    def _reward(self, obs: Observation | None = None) -> float:
        # Sparse + small shaping toward target.
        return float(-self._ee_target_error(obs))

    @property
    def joint_ranges(self) -> npt.NDArray[np.float64]:
        """Limits for the six controlled joints in API order."""
        return self.model.jnt_range[self._joint_ids].copy()

    @property
    def joint_positions(self) -> npt.NDArray[np.float64]:
        """Controlled joint positions in API order."""
        return self.data.qpos[self._qpos_adr].copy()

    @property
    def residual_physics(self) -> ResidualPhysicsConfig:
        return self._residual_physics

    @property
    def last_applied_action(self) -> npt.NDArray[np.float64]:
        return self._last_applied_action.copy()
