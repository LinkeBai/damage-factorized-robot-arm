"""Exact fixed-transform contraction for the five-joint push arm."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def translation(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(value, dtype=np.float64)
    return result


def axis_angle(axis: npt.ArrayLike, angle: float) -> npt.NDArray[np.float64]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


@dataclass(frozen=True)
class ContractedEdge:
    """Fixed transform between two consecutive dynamic joints (or chain ends)."""

    source_joint: int | None
    target_joint: int | None
    transform: npt.NDArray[np.float64]
    contracted_joints: tuple[int, ...]


class FixedTransformChain:
    """Kinematic source of truth matching ``sim/assets/arm_push.xml``."""

    axes = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64,
    )
    # Translation from the previous joint frame to each joint frame.
    origins = np.asarray(
        [[0.0, 0.0, 0.120], [0.0, 0.0, 0.0], [0.0, 0.0, 0.110],
         [0.0, 0.0, 0.120], [0.0, 0.0, 0.060]], dtype=np.float64,
    )
    tool_transform = translation([0.0, -0.0132, 0.110]) @ translation([0.020, 0.0, 0.0])

    def validate_inputs(
        self, q: npt.ArrayLike, mask: npt.ArrayLike, lock_angle: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        lock_angle = np.asarray(lock_angle, dtype=np.float64)
        if q.shape != (5,) or mask.shape != (5,) or lock_angle.shape != (5,):
            raise ValueError("q, mask and lock_angle must each have shape (5,)")
        if not set(mask.tolist()).issubset({0.0, 1.0}):
            raise ValueError("mask must be binary")
        return q, mask, lock_angle

    def effective_q(
        self, q: npt.ArrayLike, mask: npt.ArrayLike, lock_angle: npt.ArrayLike
    ) -> np.ndarray:
        q, mask, lock_angle = self.validate_inputs(q, mask, lock_angle)
        return q * (1.0 - mask) + lock_angle * mask

    def forward_pose(
        self, q: npt.ArrayLike, mask: npt.ArrayLike | None = None,
        lock_angle: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        mask = np.zeros(5) if mask is None else np.asarray(mask, dtype=np.float64)
        lock_angle = np.zeros(5) if lock_angle is None else np.asarray(lock_angle, dtype=np.float64)
        values = self.effective_q(q, mask, lock_angle)
        transform = np.eye(4, dtype=np.float64)
        for origin, axis, angle in zip(self.origins, self.axes, values):
            transform = transform @ translation(origin) @ axis_angle(axis, float(angle))
        return transform @ self.tool_transform

    def contract(
        self, mask: npt.ArrayLike, lock_angle: npt.ArrayLike
    ) -> tuple[ContractedEdge, ...]:
        """Fold locked joint rotations and fixed offsets into SE(3) edges."""
        _, mask, lock_angle = self.validate_inputs(np.zeros(5), mask, lock_angle)
        free = [index for index in range(5) if mask[index] == 0.0]
        boundaries: list[int | None] = [None, *free, None]
        edges: list[ContractedEdge] = []
        for source, target in zip(boundaries[:-1], boundaries[1:]):
            start = 0 if source is None else source + 1
            stop = 5 if target is None else target + 1
            transform = np.eye(4, dtype=np.float64)
            contracted: list[int] = []
            for joint in range(start, stop):
                transform = transform @ translation(self.origins[joint])
                if mask[joint] == 1.0:
                    transform = transform @ axis_angle(self.axes[joint], float(lock_angle[joint]))
                    contracted.append(joint)
                elif joint != target:
                    raise AssertionError("unexpected free joint inside contracted edge")
            if target is None:
                transform = transform @ self.tool_transform
            edges.append(ContractedEdge(source, target, transform, tuple(contracted)))
        return tuple(edges)

    def contracted_forward_pose(
        self, free_q: npt.ArrayLike, mask: npt.ArrayLike, lock_angle: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        mask = np.asarray(mask, dtype=np.float64)
        free = [index for index in range(5) if mask[index] == 0.0]
        values = np.asarray(free_q, dtype=np.float64)
        if values.shape != (len(free),):
            raise ValueError(f"free_q must have shape ({len(free)},)")
        transform = np.eye(4, dtype=np.float64)
        edges = self.contract(mask, lock_angle)
        for edge_index, edge in enumerate(edges):
            transform = transform @ edge.transform
            if edge.target_joint is not None:
                transform = transform @ axis_angle(
                    self.axes[edge.target_joint], float(values[edge_index])
                )
        return transform


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    if np.linalg.norm(relative - np.eye(3), ord="fro") < 1e-12:
        return 0.0
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))
