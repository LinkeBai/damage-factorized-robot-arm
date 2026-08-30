"""Robot-agnostic trajectory contract for cross-arm Push and Grasp.

Joint nodes remain variable length.  Object state is always represented as a
world-frame SE(3) pose (xyz + MuJoCo wxyz quaternion) and a twist ordered as
linear xyz then angular xyz.  A planar object is embedded in this contract
without inventing unmodelled degrees of freedom: its z/orientation remain
constant and unsupported twist components remain zero in the source simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from robotarm.envs.damage import DamageConfig


@dataclass
class VariableDofTrajectory:
    robot: str
    task: str
    joint_names: tuple[str, ...]
    joint_state: torch.Tensor  # [T+1, N, 2] = q, qvel
    actions: torch.Tensor  # [T, N]
    applied_actions: torch.Tensor  # [T, N]
    lock_mask: torch.Tensor  # [N]
    lock_angle: torch.Tensor  # [N]
    object_pose: torch.Tensor  # [T+1, 7] = xyz, wxyz
    object_twist: torch.Tensor  # [T+1, 6] = linear xyz, angular xyz
    contact_mask: torch.Tensor  # [T]
    gripper_state: torch.Tensor | None = None  # [T+1, G]
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    def validate(self, *, projection_tolerance: float = 1e-6) -> None:
        n, t = self.dof, self.steps
        expected = {
            "joint_state": (t + 1, n, 2),
            "actions": (t, n),
            "applied_actions": (t, n),
            "lock_mask": (n,),
            "lock_angle": (n,),
            "object_pose": (t + 1, 7),
            "object_twist": (t + 1, 6),
            "contact_mask": (t,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if value.dtype != torch.bool and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if not torch.all((self.lock_mask == 0) | (self.lock_mask == 1)):
            raise ValueError("lock_mask must be binary")
        quaternion_norm = torch.linalg.vector_norm(self.object_pose[:, 3:], dim=-1)
        if not torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-5):
            raise ValueError("object quaternion must be unit length")
        locked = self.lock_mask.bool()
        if locked.any():
            q_error = (self.joint_state[:, locked, 0] - self.lock_angle[locked]).abs().max()
            v_error = self.joint_state[:, locked, 1].abs().max()
            a_error = self.applied_actions[:, locked].abs().max()
            if max(float(q_error), float(v_error), float(a_error)) > projection_tolerance:
                raise ValueError("locked coordinate violates analytic trajectory contract")
        if self.gripper_state is not None:
            if self.gripper_state.ndim != 2 or self.gripper_state.shape[0] != t + 1:
                raise ValueError("gripper_state must have shape [T+1, G]")
            if not torch.isfinite(self.gripper_state).all():
                raise ValueError("gripper_state contains non-finite values")


def observe_mujoco_nodes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_names: tuple[str, ...],
    object_body: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the common joint-node, object-pose and object-twist state."""
    joint_state = np.empty((len(joint_names), 2), dtype=np.float64)
    for index, name in enumerate(joint_names):
        joint = model.joint(name)
        joint_state[index, 0] = data.qpos[int(joint.qposadr[0])]
        joint_state[index, 1] = data.qvel[int(joint.dofadr[0])]
    body = data.body(object_body)
    pose = np.concatenate([np.asarray(body.xpos), np.asarray(body.xquat)]).copy()
    # MuJoCo cvel is angular then linear; contract is linear then angular.
    cvel = np.asarray(body.cvel)
    twist = np.concatenate([cvel[3:], cvel[:3]]).copy()
    return joint_state, pose, twist


def make_single_transition(
    *,
    robot: str,
    task: str,
    joint_names: tuple[str, ...],
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray, np.ndarray],
    action: np.ndarray,
    applied_action: np.ndarray,
    damage: DamageConfig,
    contact: bool,
) -> VariableDofTrajectory:
    if damage.dof != len(joint_names):
        raise ValueError("damage DoF must match the full robot joint set")
    trajectory = VariableDofTrajectory(
        robot=robot,
        task=task,
        joint_names=joint_names,
        joint_state=torch.as_tensor(np.stack([before[0], after[0]]), dtype=torch.float32),
        actions=torch.as_tensor(np.asarray(action)[None], dtype=torch.float32),
        applied_actions=torch.as_tensor(np.asarray(applied_action)[None], dtype=torch.float32),
        lock_mask=torch.as_tensor(damage.joint_mask, dtype=torch.float32),
        lock_angle=torch.as_tensor(damage.lock_angle, dtype=torch.float32),
        object_pose=torch.as_tensor(np.stack([before[1], after[1]]), dtype=torch.float32),
        object_twist=torch.as_tensor(np.stack([before[2], after[2]]), dtype=torch.float32),
        contact_mask=torch.as_tensor([contact], dtype=torch.bool),
    )
    trajectory.validate()
    return trajectory


def collate_variable_trajectories(
    trajectories: list[VariableDofTrajectory],
) -> dict[str, torch.Tensor]:
    """Pad node/time axes with explicit masks; never expose robot identity."""
    if not trajectories:
        raise ValueError("trajectories cannot be empty")
    for trajectory in trajectories:
        trajectory.validate()
    batch = len(trajectories)
    max_steps = max(t.steps for t in trajectories)
    max_dof = max(t.dof for t in trajectories)
    joint_state = torch.zeros(batch, max_steps + 1, max_dof, 2)
    actions = torch.zeros(batch, max_steps, max_dof)
    applied = torch.zeros_like(actions)
    lock_mask = torch.zeros(batch, max_dof)
    lock_angle = torch.zeros(batch, max_dof)
    object_pose = torch.zeros(batch, max_steps + 1, 7)
    object_twist = torch.zeros(batch, max_steps + 1, 6)
    contact = torch.zeros(batch, max_steps, dtype=torch.bool)
    node_valid = torch.zeros(batch, max_dof, dtype=torch.bool)
    transition_valid = torch.zeros(batch, max_steps, dtype=torch.bool)
    for row, trajectory in enumerate(trajectories):
        t, n = trajectory.steps, trajectory.dof
        joint_state[row, : t + 1, :n] = trajectory.joint_state
        actions[row, :t, :n] = trajectory.actions
        applied[row, :t, :n] = trajectory.applied_actions
        lock_mask[row, :n] = trajectory.lock_mask
        lock_angle[row, :n] = trajectory.lock_angle
        object_pose[row, : t + 1] = trajectory.object_pose
        object_twist[row, : t + 1] = trajectory.object_twist
        contact[row, :t] = trajectory.contact_mask
        node_valid[row, :n] = True
        transition_valid[row, :t] = True
    return {
        "joint_state": joint_state,
        "actions": actions,
        "applied_actions": applied,
        "lock_mask": lock_mask,
        "lock_angle": lock_angle,
        "object_pose": object_pose,
        "object_twist": object_twist,
        "contact_mask": contact,
        "node_valid": node_valid,
        "transition_valid": transition_valid,
    }
