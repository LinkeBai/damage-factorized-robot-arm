"""Damage configuration for a deployment (PROJECT-PLAN-V4 §4.x, G0).

Concretely implements the ``DamageConfig`` protocol declared in
``protocol.py``: a binary ``joint_mask`` (1 = joint locked) and an absolute
``lock_angle`` per joint. Used both to seed the sim (lock a joint at a fixed
angle) and to describe the real arm during calibration/evaluation.

A locked joint is held rigid at ``lock_angle[i]``; a joint that is *not*
locked still carries an entry in ``lock_angle`` but the value is ignored.
"""
from __future__ import annotations

from functools import total_ordering
from typing import Literal

import numpy as np
import numpy.typing as npt


@total_ordering
class DamageConfig:
    """Model of which joints are locked and at what angle.

    Parameters
    ----------
    joint_mask:
        Binary array of shape ``(N_JOINTS,)``; ``1`` means the joint is
        locked/immobilized, ``0`` means it is free and actuated.
    lock_angle:
        Absolute angle (rad) per joint, shape ``(N_JOINTS,)``. Only entries
        where ``joint_mask == 1`` are meaningful.
    dof:
        Number of positioning joints (defaults to 5). Must match the
        length of the arrays.

    Raises
    ------
    ValueError
        If the mask is non-binary or the array lengths do not match ``dof``.
    """

    __slots__ = ("joint_mask", "lock_angle", "dof")

    def __init__(
        self,
        joint_mask: npt.ArrayLike,
        lock_angle: npt.ArrayLike,
        dof: int = 5,
    ) -> None:
        mask = np.asarray(joint_mask, dtype=np.int64)
        angle = np.asarray(lock_angle, dtype=np.float64)

        if mask.ndim != 1 or mask.shape[0] != dof:
            raise ValueError(f"joint_mask must be 1-D of length {dof}, got {mask.shape}")
        if angle.ndim != 1 or angle.shape[0] != dof:
            raise ValueError(f"lock_angle must be 1-D of length {dof}, got {angle.shape}")
        if not set(mask.tolist()).issubset({0, 1}):
            raise ValueError("joint_mask entries must be 0 or 1")

        self.joint_mask = mask
        self.lock_angle = angle
        self.dof = dof

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def intact(cls, dof: int = 5) -> "DamageConfig":
        """No joint locked — the healthy / baseline morphology."""
        return cls(np.zeros(dof, dtype=np.int64), np.zeros(dof), dof=dof)

    @classmethod
    def lock_single(cls, joint: int, angle: float, dof: int = 5) -> "DamageConfig":
        """Lock exactly one joint ``joint`` at ``angle`` (rad)."""
        if not 0 <= joint < dof:
            raise ValueError(f"joint index {joint} out of range for dof={dof}")
        mask = np.zeros(dof, dtype=np.int64)
        mask[joint] = 1
        angles = np.zeros(dof)
        angles[joint] = float(angle)
        return cls(mask, angles, dof=dof)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def locked(self) -> list[int]:
        """Indices of the locked joints."""
        return [int(i) for i in np.flatnonzero(self.joint_mask)]

    @property
    def n_locked(self) -> int:
        return int(self.joint_mask.sum())

    def lock_angle_of(self, joint: int) -> float:
        """Return the lock angle for ``joint`` (meaningful only if locked)."""
        return float(self.lock_angle[joint])

    def as_protocol(self) -> "DamageConfig":
        """Return self — this type already satisfies the DamageConfig protocol."""
        return self

    def copy(self) -> "DamageConfig":
        return DamageConfig(self.joint_mask.copy(), self.lock_angle.copy(), dof=self.dof)

    # ------------------------------------------------------------------
    # Value semantics (equality + order) for run naming / dedup
    # ------------------------------------------------------------------
    def _key(self) -> tuple:
        return (tuple(self.joint_mask.tolist()), tuple(np.round(self.lock_angle, 9).tolist()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DamageConfig):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: "DamageConfig") -> bool:
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"DamageConfig(mask={self.joint_mask.tolist()}, "
            f"lock={self.lock_angle.tolist()})"
        )


# Pre-built canonical failure modes used across the plan's task splits.
# D1..D4 label the damaged deployments in the experiments (D2/D3 are the ones
# G0 must show share a reachable region).
D0 = DamageConfig.intact


def D1(dof: int = 5) -> DamageConfig:
    return DamageConfig.lock_single(0, 0.0, dof=dof)


def D2(dof: int = 5) -> DamageConfig:
    return DamageConfig.lock_single(1, 0.5, dof=dof)


def D3(dof: int = 5) -> DamageConfig:
    return DamageConfig.lock_single(2, -0.5, dof=dof)


def D4(dof: int = 5) -> DamageConfig:
    return DamageConfig.lock_single(3, 0.9, dof=dof)


def make_damage(style: str | Literal["intact", "D1", "D2", "D3", "D4"], dof: int = 5) -> DamageConfig:
    """Build a canonical damage config by name (used by configs/tests)."""
    table = {"intact": D0, "D1": D1, "D2": D2, "D3": D3, "D4": D4}
    try:
        return table[style](dof)
    except KeyError:
        raise KeyError(f"unknown damage style: {style!r}; choose from {sorted(table)}") from None
