"""Configurable full-joint MuJoCo interface for cross-robot data collection.

The learned model sees normalized per-joint commands only.  This environment
keeps unavoidable low-level actuator differences outside the model: torque
actuators receive normalized force commands, while position actuators receive
a bounded target increment from the current joint position.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robotarm.envs.damage import DamageConfig
from robotarm.training.variable_trajectory import observe_mujoco_nodes


class VariableMujocoArmEnv:
    def __init__(
        self,
        xml_path: str | Path,
        *,
        joint_names: tuple[str, ...],
        actuator_names: tuple[str, ...],
        object_body: str,
        object_geom: str,
        home_keyframe: str | None = None,
        position_increment_rad: float = 0.05,
    ) -> None:
        if len(joint_names) != len(actuator_names):
            raise ValueError("joint and actuator sets must have equal length")
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self.object_body = object_body
        self.object_geom = object_geom
        self.home_keyframe = home_keyframe
        self.position_increment_rad = float(position_increment_rad)
        self.joint_ids = np.array([self.model.joint(x).id for x in joint_names], dtype=int)
        self.actuator_ids = np.array([self.model.actuator(x).id for x in actuator_names], dtype=int)
        self.qpos_adrs = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adrs = self.model.jnt_dofadr[self.joint_ids]
        self.damage = DamageConfig.intact(len(joint_names))
        self.last_applied_action = np.zeros(len(joint_names), dtype=np.float64)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def reset(self, damage: DamageConfig | None = None):
        self.damage = damage.copy() if damage is not None else DamageConfig.intact(self.dof)
        if self.damage.dof != self.dof:
            raise ValueError("damage DoF must match full controlled joint set")
        if self.home_keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, self.model.key(self.home_keyframe).id
            )
        self._apply_damage()
        mujoco.mj_forward(self.model, self.data)
        self.last_applied_action[:] = 0.0
        return self.observe()

    def observe(self):
        return observe_mujoco_nodes(
            self.model, self.data,
            joint_names=self.joint_names, object_body=self.object_body,
        )

    def _apply_damage(self) -> None:
        for joint in self.damage.locked:
            self.data.qpos[self.qpos_adrs[joint]] = self.damage.lock_angle_of(joint)
            self.data.qvel[self.dof_adrs[joint]] = 0.0

    def _map_action(self, normalized: np.ndarray) -> None:
        for index, actuator_id in enumerate(self.actuator_ids):
            actuator = self.model.actuator(int(actuator_id))
            low, high = np.asarray(actuator.ctrlrange, dtype=np.float64)
            bias_type = int(self.model.actuator_biastype[actuator_id])
            if bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE):
                target = self.data.qpos[self.qpos_adrs[index]] + (
                    self.position_increment_rad * normalized[index]
                )
                self.data.ctrl[actuator_id] = np.clip(target, low, high)
            else:
                scale = max(abs(float(low)), abs(float(high)))
                self.data.ctrl[actuator_id] = np.clip(normalized[index] * scale, low, high)

    def step(self, action: np.ndarray):
        normalized = np.asarray(action, dtype=np.float64).reshape(-1)
        if normalized.shape != (self.dof,):
            raise ValueError(f"action must have shape ({self.dof},)")
        normalized = np.clip(normalized, -1.0, 1.0)
        normalized[self.damage.locked] = 0.0
        self._map_action(normalized)
        mujoco.mj_step(self.model, self.data)
        self._apply_damage()
        mujoco.mj_forward(self.model, self.data)
        self.last_applied_action = normalized.copy()
        return self.observe()

    def object_robot_contact(self) -> bool:
        object_id = int(self.model.geom(self.object_geom).id)
        excluded = {
            int(self.model.geom(name).id)
            for name in ("table_geom", "task_table")
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        }
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if object_id in pair and not pair.intersection(excluded):
                return True
        return False
