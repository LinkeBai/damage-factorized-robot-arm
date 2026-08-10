"""Hardware-agnostic safety layer (PROJECT-PLAN-V4 §11, G0 §7).

The plan requires every deployment (sim or real) to enforce safe behaviour:
joint soft limits, a maximum commanded speed, a maximum control input, and an
emergency stop that must *always* work. A locked joint must never be commanded
to move. Policy/adaptation code talks only to these helpers through the
protocol, so the same checks run in simulation and on the real arm.

Thresholds are deliberately loaded at construction from a config mapping and
are *not* hard-coded here, because the plan forbids inventing fixed limit
values before G0 measurement ("阈值必须来自数据手册或 G0 实测"). The defaults
below are engineering all-zero placeholders that force the caller to supply
real values from ``hardware/safety_limits.yaml`` once measured.

Safety verdicts are returned as (ok, reason) pairs so callers can record them
in ``safety_flags`` / ``hardware_state`` without exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import numpy.typing as npt

__all__ = ["SafetyLimits", "SafetyViolation", "SafetyMonitor"]


@dataclass(frozen=True)
class SafetyViolation:
    """A single safety check that failed."""

    code: str  # machine-readable, e.g. 'joint_limit'
    joint: int | None
    value: float
    limit: float
    message: str

    def to_flag(self) -> dict[str, bool]:
        return {f"safety_{self.code}": True}


@dataclass
class SafetyLimits:
    """Limit set for one deployment; maps to ``hardware/safety_limits.yaml``.

    ``joint_range`` has shape (dof, 2) = [min, max] per joint, rad.
    ``max_joint_speed`` and ``max_ctrl`` are length-dof arrays.
    ``max_joint_speed`` names are per-joint; a scalar broadcast is allowed.
    """

    dof: int
    joint_range: npt.NDArray  # (dof, 2), rad
    max_joint_speed: npt.NDArray  # (dof,) rad/s
    max_ctrl: npt.NDArray  # (dof,) normalized [-1,1] input cap
    max_command_delta: npt.NDArray | None = field(default=None)  # (dof,) per-step

    @classmethod
    def from_mapping(cls, cfg: Mapping, dof: int) -> "SafetyLimits":
        joint_range = np.asarray(cfg["joint_range"], dtype=np.float64).reshape(dof, 2)
        max_joint_speed = np.asarray(
            cfg.get("max_joint_speed", np.zeros(dof)), dtype=np.float64
        ).reshape(-1)
        max_ctrl = np.asarray(cfg.get("max_ctrl", np.zeros(dof)), dtype=np.float64).reshape(-1)
        if max_joint_speed.size == 1:
            max_joint_speed = np.full(dof, float(max_joint_speed[0]))
        if max_ctrl.size == 1:
            max_ctrl = np.full(dof, float(max_ctrl[0]))
        max_command_delta = None
        if "max_command_delta" in cfg:
            v = np.asarray(cfg["max_command_delta"], dtype=np.float64).reshape(-1)
            max_command_delta = np.full(dof, float(v[0])) if v.size == 1 else v
        return cls(
            dof=dof,
            joint_range=joint_range,
            max_joint_speed=max_joint_speed,
            max_ctrl=max_ctrl,
            max_command_delta=max_command_delta,
        )


class SafetyMonitor:
    """Incremental safety checks over commands and joint states.

    Parameters
    ----------
    limits:
        The limit set for this deployment.
    locked_joints:
        Indices of joints that are locked and must never be commanded to move.
    """

    def __init__(self, limits: SafetyLimits, locked_joints: list[int] | None = None) -> None:
        self.limits = limits
        self.locked = set(locked_joints or [])
        self._last_ctrl: npt.NDArray | None = None

    # ------------------------------------------------------------------
    # Command checks (before sending to the arm)
    # ------------------------------------------------------------------
    def check_ctrl(
        self, ctrl: npt.ArrayLike, prev_ctrl: npt.ArrayLike | None = None
    ) -> tuple[bool, list[SafetyViolation]]:
        """Validate a normalized command. Never command a locked joint."""
        ctrl = np.asarray(ctrl, dtype=np.float64)
        violations: list[SafetyViolation] = []
        if ctrl.shape != (self.limits.dof,):
            violations.append(
                SafetyViolation("ctrl_dim", None, len(ctrl), self.limits.dof, "bad command dim")
            )
            return (False, violations)

        for j in self.locked:
            if abs(ctrl[j]) > 0.0:
                violations.append(
                    SafetyViolation("locked_joint_command", j, ctrl[j], 0.0, "command sent to locked joint")
                )
        if np.any(np.abs(ctrl) > self.limits.max_ctrl):
            j = int(np.argmax(np.abs(ctrl)))
            violations.append(
                SafetyViolation("ctrl_limit", j, float(ctrl[j]), float(self.limits.max_ctrl[j]), "ctrl exceeds cap")
            )
        if (
            self.limits.max_command_delta is not None
            and prev_ctrl is not None
        ):
            delta = np.abs(ctrl - np.asarray(prev_ctrl))
            if np.any(delta > self.limits.max_command_delta):
                j = int(np.argmax(delta))
                violations.append(
                    SafetyViolation("command_delta", j, float(delta[j]), float(self.limits.max_command_delta[j]), "command jump")
                )
        return (not violations, violations)

    # ------------------------------------------------------------------
    # State checks (on observed joint positions/velocities)
    # ------------------------------------------------------------------
    def check_state(
        self, qpos: npt.ArrayLike, qvel: npt.ArrayLike
    ) -> tuple[bool, list[SafetyViolation]]:
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        violations: list[SafetyViolation] = []
        for j in range(self.limits.dof):
            lo, hi = self.limits.joint_range[j]
            if not lo <= qpos[j] <= hi:
                which = "below" if qpos[j] < lo else "above"
                violations.append(
                    SafetyViolation("joint_limit", j, float(qpos[j]), float(hi if qpos[j] > hi else lo), f"joint {j} {which} range")
                )
            lim = self.limits.max_joint_speed[j]
            if abs(qvel[j]) > lim:
                violations.append(
                    SafetyViolation("joint_speed", j, float(qvel[j]), float(lim), f"joint {j} speed")
                )
        return (not violations, violations)

    # ------------------------------------------------------------------
    # Convenience: gate a command, returning a safe zeroed fallback on breach
    # ------------------------------------------------------------------
    def gate(self, ctrl: npt.ArrayLike) -> tuple[npt.NDArray, bool, list[SafetyViolation]]:
        """Return (safe_ctrl, was_accepted, violations).

        If any check fails, returns a zero command and ``was_accepted=False``
        so the caller can stop rather than send a dangerous input.
        """
        ctrl = np.asarray(ctrl, dtype=np.float64)
        # First clamp any step that would exceed the max per-command delta.
        if self.limits.max_command_delta is not None and self._last_ctrl is not None:
            ctrl = np.clip(ctrl, self._last_ctrl - self.limits.max_command_delta,
                           self._last_ctrl + self.limits.max_command_delta)
        ok, violations = self.check_ctrl(ctrl, self._last_ctrl)
        if ok:
            self._last_ctrl = ctrl.copy()
        else:
            self._last_ctrl = None
        return (ctrl if ok else np.zeros_like(ctrl), ok, violations)

    def reset_tracking(self) -> None:
        self._last_ctrl = None