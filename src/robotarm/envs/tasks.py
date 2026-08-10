"""Task definitions (PROJECT-PLAN-V4 §5): Reach, Push, and conditional Pick.

A task is a small object that, given the current end-effector / object pose
and a target, decides success and reward. It is decoupled from any specific
environment so the same definition drives sim, real, calibration, validation
and evaluation. Task constants that depend on physical measurements
(tolerance, object size) are injection points to be finalized at G0 — the
plan forbids inventing final numbers before measurement ("阈值必须来自数据手册
或 G0 实测").

The reachable/valid target set must be drawn from the *common* reachable
region of intact and damaged morphologies (G0 §4), so target sampling lives
with the reachability utilities rather than here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

__all__ = ["Task", "ReachTask", "PushTask", "PickTask", "SuccessFn", "RewardFn", "build_task"]

# SuccessFn(relevant_state, target) -> bool ; RewardFn -> float
SuccessFn = Callable[[dict[str, npt.NDArray], npt.NDArray], bool]
RewardFn = Callable[[dict[str, npt.NDArray], npt.NDArray], float]


class Task(ABC):
    """Base task: a success predicate plus a reward over task-relevant state."""

    task_id: str

    @abstractmethod
    def success(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> bool: ...

    @abstractmethod
    def reward(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> float: ...


def _vec(state: dict[str, npt.NDArray], key: str) -> npt.NDArray:
    return np.asarray(state[key], dtype=np.float64)


def _dist(state: dict[str, npt.NDArray], key: str, target: npt.NDArray) -> float:
    return float(np.linalg.norm(_vec(state, key) - target))


class ReachTask(Task):
    """§5.1: move the end-effector tip to a 3D target.

    ``ee_pos`` in ``state`` is compared to ``target`` (3-D). Success when the
    distance is within ``tolerance`` (default 0.05 m — the plan's engineering
    starting point, to be recalibrated at G0 using camera / kinematics error).
    """

    task_id = "reach"

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def success(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> bool:
        return _dist(state, "ee_pos", target) <= self.tolerance

    def reward(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> float:
        d = _dist(state, "ee_pos", target)
        return -d + 5.0 * (1.0 if d <= self.tolerance else 0.0)


class PushTask(Task):
    """§5.2: push the object centroid into the goal region.

    ``obj_pos`` (3-D) is the object centroid probed from state; ``target`` is
    the goal position. Requires fixed object size, friction surface and camera
    calibration before use (per §5.2).
    """

    task_id = "push"

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def success(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> bool:
        return _dist(state, "obj_pos", target) <= self.tolerance

    def reward(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> float:
        d = _dist(state, "obj_pos", target)
        return -d + 5.0 * (1.0 if d <= self.tolerance else 0.0)


class PickTask(Task):
    """§5.3 (conditional): grasp and lift the object toward ``target``.

    Only enters the experiment set once the position-only reachability gate
    passes and the gripper / object detection are stable. If the gate fails
    the task is dropped rather than explaining an adaptation failure (plan
    §5.3). ``state["grasped"]`` must be a bool.
    """

    task_id = "pick"

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def success(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> bool:
        grasped = bool(state["grasped"])
        return grasped and _dist(state, "obj_pos", target) <= self.tolerance

    def reward(self, state: dict[str, npt.NDArray], target: npt.NDArray) -> float:
        d = _dist(state, "obj_pos", target)
        terminal = 1.0 if self.success(state, target) else 0.0
        return -d + 10.0 * terminal


_TASK_TABLE = {"reach": ReachTask, "push": PushTask, "pick": PickTask}


def build_task(kind: str, **kwargs: Any) -> Task:
    """Factory: ``build_task('reach', tolerance=0.06)``."""
    try:
        return _TASK_TABLE[kind](**kwargs)
    except KeyError:
        raise KeyError(f"unknown task kind {kind!r}; choose from {sorted(_TASK_TABLE)}") from None