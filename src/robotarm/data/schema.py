"""Trajectory / episode data schema (PROJECT-PLAN-V4 §10.1).

Defines the canonical, immutable record for one episode and per-step fields.
Raw trajectories are append-only and never edited in place (§10.2); cleaning
produces a new dataset version with its own manifest. This module only
declares the schema and validates records — it performs no I/O so that it can
be imported from anywhere (envs, training, analysis, tests) without side
effects.

Field names are written in snake_case and match §10.1 verbatim so that the
plan's audit trail maps 1:1 onto the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Enumerations (kept as plain strings; validated at write time, per plan §10.2)
# ---------------------------------------------------------------------------
Platform = Literal["sim", "real"]
Split = Literal["calibration", "validation", "evaluation"]

# Five positioning joints. The gripper-open channel is separate.
N_JOINTS = 5

EPISODE_FIELDS = (
    "episode_id",
    "timestamp_ns",
    "platform",
    "task_id",
    "target_id",
    "split",
    "damage_id",
    "joint_mask",
    "lock_angle",
    "observation",
    "action_commanded",
    "action_applied",
    "next_observation",
    "reward",
    "success",
    "done",
    "safety_flags",
    "hardware_state",
    "camera_frame_ref",
    "config_hash",
    "git_commit",
    "seed",
)


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    """The mutable per-step fields that accumulate into an episode.

    ``observation`` / ``next_observation`` are dict-of-arrays matching the
    :class:`~robotarm.envs.protocol.Observation` convention (``state``,
    ``target``, optional ``image*``). ``action_commanded`` is what the policy
    asked for; ``action_applied`` is what the environment actually sent to the
    actuators (identical for sim, may differ on real hardware after clamping).
    """

    observation: dict[str, npt.NDArray]
    action_commanded: npt.NDArray[np.float64]
    action_applied: npt.NDArray[np.float64]
    next_observation: dict[str, npt.NDArray]
    reward: float
    success: bool
    done: bool
    safety_flags: dict[str, bool] = field(default_factory=dict)
    hardware_state: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    """One complete, immutable episode conforming to §10.1.

    ``steps`` holds the per-step records; the scalar/identity fields live
    directly on the episode. ``joint_mask``/``lock_angle`` are the damage
    configuration (may be all-zeros for an intact baseline episode).
    """

    episode_id: str
    timestamp_ns: int
    platform: Platform
    task_id: str
    target_id: str
    split: Split
    damage_id: str
    joint_mask: npt.NDArray[np.int64]
    lock_angle: npt.NDArray[np.float64]
    steps: list[StepRecord]
    seed: int
    config_hash: str = ""
    git_commit: str = ""
    camera_frame_ref: str | None = None

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` if this episode does not conform to §10.1."""
        if self.platform not in ("sim", "real"):
            raise ValueError(f"invalid platform: {self.platform!r}")
        if self.split not in ("calibration", "validation", "evaluation"):
            raise ValueError(f"invalid split: {self.split!r}")
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        _validate_damage(self.joint_mask, self.lock_angle)
        if not self.steps:
            raise ValueError("episode must contain at least one step")
        _validate_step(self.steps[0])
        _validate_step(self.steps[-1])
        for step in self.steps:
            if step.observation.keys() != self.steps[0].observation.keys():
                raise ValueError("observation key sets differ across steps")

    # ------------------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def n_transitions(self) -> int:
        """Number of (s, a, s') transitions; requires a terminal ``done``."""
        return len(self.steps) - 1

    # ------------------------------------------------------------------
    def to_tree(self) -> dict[str, Any]:
        """Flatten to a nested dict (observations kept as arrays).

        Arrays are returned by reference; callers that mutate must deep-copy.
        """
        return {
            "episode_id": self.episode_id,
            "timestamp_ns": self.timestamp_ns,
            "platform": self.platform,
            "task_id": self.task_id,
            "target_id": self.target_id,
            "split": self.split,
            "damage_id": self.damage_id,
            "joint_mask": self.joint_mask,
            "lock_angle": self.lock_angle,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "camera_frame_ref": self.camera_frame_ref,
            "steps": [
                {
                    "observation": s.observation,
                    "action_commanded": s.action_commanded,
                    "action_applied": s.action_applied,
                    "next_observation": s.next_observation,
                    "reward": s.reward,
                    "success": s.success,
                    "done": s.done,
                    "safety_flags": s.safety_flags,
                    "hardware_state": s.hardware_state,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_damage(joint_mask: npt.NDArray, lock_angle: npt.NDArray) -> None:
    joint_mask = np.asarray(joint_mask)
    lock_angle = np.asarray(lock_angle, dtype=np.float64)
    if joint_mask.shape != (N_JOINTS,):
        raise ValueError(f"joint_mask must have shape ({N_JOINTS},), got {joint_mask.shape}")
    if lock_angle.shape != (N_JOINTS,):
        raise ValueError(f"lock_angle must have shape ({N_JOINTS},), got {lock_angle.shape}")
    if not set(np.asarray(joint_mask).tolist()).issubset({0, 1}):
        raise ValueError("joint_mask entries must be 0 or 1")


def _validate_step(step: StepRecord) -> None:
    if "state" not in step.observation:
        raise ValueError("observation must contain 'state' (see observation protocol)")
    expected_state = (2 * N_JOINTS,)
    if np.asarray(step.observation["state"]).shape != expected_state:
        raise ValueError(f"observation state must have shape {expected_state}")
    if np.asarray(step.next_observation["state"]).shape != expected_state:
        raise ValueError(f"next observation state must have shape {expected_state}")
    if step.action_commanded.shape != (N_JOINTS,):
        raise ValueError(f"action_commanded must have shape ({N_JOINTS},)")
    if step.action_applied.shape != (N_JOINTS,):
        raise ValueError(f"action_applied must have shape ({N_JOINTS},)")
