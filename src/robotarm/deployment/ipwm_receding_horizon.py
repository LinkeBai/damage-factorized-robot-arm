"""Auditable receding-horizon adapter for real IPWM deployment.

This module deliberately contains no hardware writes.  It turns a fresh joint
and object observation into a newly scored bank of short joint-reference
sequences.  The hardware runner may execute only the first reference returned
by :meth:`plan`, then it must acquire a new observation and call again.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable

import numpy as np


ScoreFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RecedingPlan:
    cycle: int
    observation_px: tuple[float, float]
    observation_monotonic_ns: int
    selected_index: int
    selected_first_reference: np.ndarray
    selected_references: np.ndarray
    scores: np.ndarray
    candidates_sha256: str
    inference_s: float
    remaining_fraction: float
    candidates: np.ndarray
    selection_eligible: np.ndarray
    fault_mask: np.ndarray
    lock_angles: np.ndarray


class IPWMRecedingHorizonPlanner:
    """Re-score a fixed, preregistered motion family after every observation.

    ``reference_deltas`` has shape ``(N,H,5)`` and is expressed relative to
    the measured joint state at each replanning cycle.  Candidate amplitudes
    contract with measured remaining image-axis error.  Locked coordinates are
    always zeroed before scoring and restored to their measured lock angles.
    """

    def __init__(
        self,
        reference_deltas: np.ndarray,
        *,
        task_start_px: tuple[float, float],
        task_goal_px: tuple[float, float],
        locked_indices: tuple[int, ...],
        score_function: ScoreFunction,
        metres_per_pixel: float,
        base_xy_per_pixel: np.ndarray,
        base_xy_intercept_m: np.ndarray,
        joint_ranges_rad: np.ndarray | None = None,
        maximum_first_step_rad: float | None = None,
        base_eligible: np.ndarray | None = None,
        execution_reference_index: int = 0,
        minimum_remaining_fraction: float = 0.05,
        pixel_to_base_affine: np.ndarray | None = None,
    ) -> None:
        deltas = np.asarray(reference_deltas, dtype=np.float64)
        if deltas.ndim != 3 or deltas.shape[2] != 5 or len(deltas) < 1:
            raise ValueError("reference_deltas must have shape (N,H,5)")
        if deltas.shape[1] < 1:
            raise ValueError("planning horizon must be positive")
        if not 0 <= execution_reference_index < deltas.shape[1]:
            raise ValueError("execution_reference_index must be inside the horizon")
        self.execution_reference_index = execution_reference_index
        if not np.isfinite(minimum_remaining_fraction) or not 0 < minimum_remaining_fraction <= 1:
            raise ValueError("minimum_remaining_fraction must be in (0,1]")
        self.minimum_remaining_fraction = float(minimum_remaining_fraction)
        axis = np.asarray(task_goal_px, dtype=float) - np.asarray(task_start_px, dtype=float)
        length = float(np.linalg.norm(axis))
        if length <= 0:
            raise ValueError("task start and goal must differ")
        self.reference_deltas = deltas.copy()
        self.task_start_px = np.asarray(task_start_px, dtype=float)
        self.task_goal_px = np.asarray(task_goal_px, dtype=float)
        self.task_axis_px = axis / length
        self.task_length_px = length
        self.locked_indices = tuple(int(i) for i in locked_indices)
        self.score_function = score_function
        self.metres_per_pixel = float(metres_per_pixel)
        self.base_xy_per_pixel = np.asarray(base_xy_per_pixel, dtype=float)
        self.base_xy_intercept_m = np.asarray(base_xy_intercept_m, dtype=float)
        # Columns are image u, image v and the intercept; calibration provenance
        # and acceptance must be checked by the caller before hardware use.
        self.pixel_to_base_affine = None
        if pixel_to_base_affine is not None:
            affine = np.asarray(pixel_to_base_affine, dtype=float)
            if (affine.shape != (2, 3) or not np.all(np.isfinite(affine))
                    or np.linalg.matrix_rank(affine[:, :2]) != 2):
                raise ValueError("pixel_to_base_affine requires a finite full-rank (2,3) matrix")
            self.pixel_to_base_affine = affine.copy()
        self.joint_ranges_rad = (
            None if joint_ranges_rad is None else np.asarray(joint_ranges_rad, dtype=float)
        )
        if self.joint_ranges_rad is not None and self.joint_ranges_rad.shape != (5, 2):
            raise ValueError("joint_ranges_rad must have shape (5,2)")
        self.maximum_first_step_rad = maximum_first_step_rad
        self.base_eligible = (
            np.ones(len(deltas), dtype=bool)
            if base_eligible is None else np.asarray(base_eligible, dtype=bool)
        )
        if self.base_eligible.shape != (len(deltas),):
            raise ValueError("base_eligible must match candidate count")
        self.cycle = 0

    def plan(
        self,
        *,
        joint_q: np.ndarray,
        joint_qd: np.ndarray,
        object_px: tuple[float, float],
        observation_monotonic_ns: int,
        lock_angles: np.ndarray,
    ) -> RecedingPlan:
        q = np.asarray(joint_q, dtype=float)
        qd = np.asarray(joint_qd, dtype=float)
        px = np.asarray(object_px, dtype=float)
        lock_angles = np.asarray(lock_angles, dtype=float)
        if q.shape != (5,) or qd.shape != (5,) or lock_angles.shape != (5,):
            raise ValueError("joint_q, joint_qd and lock_angles must have shape (5,)")
        if not np.all(np.isfinite(np.r_[q, qd, px, lock_angles])):
            raise ValueError("closed-loop observation must be finite")

        remaining_px = float(np.dot(self.task_goal_px - px, self.task_axis_px))
        remaining_fraction = float(np.clip(remaining_px / self.task_length_px,
                                          self.minimum_remaining_fraction, 1.25))
        candidates = q[None, None, :] + remaining_fraction * self.reference_deltas
        for index in self.locked_indices:
            candidates[:, :, index] = lock_angles[index]

        if self.pixel_to_base_affine is None:
            object_xy = px[0] * self.base_xy_per_pixel + self.base_xy_intercept_m
            goal_xy = self.task_goal_px[0] * self.base_xy_per_pixel + self.base_xy_intercept_m
        else:
            object_xy = self.pixel_to_base_affine @ np.r_[px, 1.0]
            goal_xy = self.pixel_to_base_affine @ np.r_[self.task_goal_px, 1.0]
        initial = np.r_[q, qd, object_xy, np.zeros(2)]
        mask = np.zeros(5, dtype=float)
        angle = np.zeros(5, dtype=float)
        for index in self.locked_indices:
            mask[index] = 1.0
            angle[index] = lock_angles[index]

        started = time.perf_counter()
        scores = np.asarray(
            self.score_function(initial, candidates, goal_xy, mask, angle), dtype=float
        )
        inference_s = time.perf_counter() - started
        if scores.shape != (len(candidates),) or not np.all(np.isfinite(scores)):
            raise RuntimeError("IPWM returned invalid online candidate scores")
        eligible = self.base_eligible.copy()
        if self.joint_ranges_rad is not None:
            eligible &= np.all(
                (candidates >= self.joint_ranges_rad[None, None, :, 0])
                & (candidates <= self.joint_ranges_rad[None, None, :, 1]),
                axis=(1, 2),
            )
        if self.maximum_first_step_rad is not None:
            eligible &= np.max(np.abs(candidates[:, self.execution_reference_index] - q[None, :]), axis=1) <= float(
                self.maximum_first_step_rad
            )
        if not np.any(eligible):
            raise RuntimeError("no online IPWM candidate passes current-state safety gates")
        selection_scores = np.where(eligible, scores, np.inf)
        selected = int(np.argmin(selection_scores))
        digest = hashlib.sha256(np.ascontiguousarray(candidates).tobytes()).hexdigest()
        result = RecedingPlan(
            cycle=self.cycle,
            observation_px=(float(px[0]), float(px[1])),
            observation_monotonic_ns=int(observation_monotonic_ns),
            selected_index=selected,
            selected_first_reference=candidates[selected, self.execution_reference_index].copy(),
            selected_references=candidates[selected].copy(),
            scores=scores.copy(),
            candidates_sha256=digest,
            inference_s=float(inference_s),
            remaining_fraction=remaining_fraction,
            candidates=candidates,
            selection_eligible=eligible.copy(),
            fault_mask=mask.copy(),
            lock_angles=angle.copy(),
        )
        self.cycle += 1
        return result
