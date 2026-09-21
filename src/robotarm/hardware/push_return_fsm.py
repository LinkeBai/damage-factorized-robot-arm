"""Guarded state machine for the supervised autonomous Push-return cycle.

This module contains no hardware side effects.  The real executor may issue a
motion command only when :meth:`PushReturnFSM.advance` returns a motion state.
Every failed observation transitions to ABORT and must be preserved in the
cycle manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CycleState(str, Enum):
    WAIT_START = "WAIT_START"
    SCENE_CLEAR = "SCENE_CLEAR"
    FORWARD_PUSH = "FORWARD_PUSH"
    FORWARD_ASSESS = "FORWARD_ASSESS"
    RETRACT = "RETRACT"
    MOVE_BEHIND = "MOVE_BEHIND"
    DESCEND = "DESCEND"
    BACK_PUSH = "BACK_PUSH"
    HOME = "HOME"
    AUDIT = "AUDIT"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class Observation:
    cube_detected: bool = True
    camera_fresh: bool = True
    feedback_ok: bool = True
    electrical_ok: bool = True
    scene_clear: bool = True
    start_stable: bool = False
    forward_finished: bool = False
    outcome_available: bool = False
    retract_finished: bool = False
    behind_clearance_ok: bool = False
    descend_finished: bool = False
    reset_stable: bool = False
    home_finished: bool = False
    audit_passed: bool = False
    elapsed_motion_s: float = 0.0


_NEXT = {
    CycleState.WAIT_START: CycleState.SCENE_CLEAR,
    CycleState.SCENE_CLEAR: CycleState.FORWARD_PUSH,
    CycleState.FORWARD_PUSH: CycleState.FORWARD_ASSESS,
    CycleState.FORWARD_ASSESS: CycleState.RETRACT,
    CycleState.RETRACT: CycleState.MOVE_BEHIND,
    CycleState.MOVE_BEHIND: CycleState.DESCEND,
    CycleState.DESCEND: CycleState.BACK_PUSH,
    CycleState.BACK_PUSH: CycleState.HOME,
    CycleState.HOME: CycleState.AUDIT,
    CycleState.AUDIT: CycleState.COMPLETE,
}


class PushReturnFSM:
    """Deterministic safety-gated lifecycle for exactly one cycle."""

    def __init__(self, maximum_motion_s: float = 10.0) -> None:
        if maximum_motion_s <= 0:
            raise ValueError("maximum_motion_s must be positive")
        self.maximum_motion_s = maximum_motion_s
        self.state = CycleState.WAIT_START
        self.abort_reason: str | None = None

    def _abort(self, reason: str) -> CycleState:
        self.abort_reason = reason
        self.state = CycleState.ABORT
        return self.state

    def advance(self, obs: Observation) -> CycleState:
        if self.state in {CycleState.COMPLETE, CycleState.ABORT}:
            return self.state
        if obs.elapsed_motion_s > self.maximum_motion_s:
            return self._abort("motion_time_budget_exceeded")
        if not obs.camera_fresh or not obs.cube_detected:
            return self._abort("cube_observation_lost")
        if not obs.feedback_ok:
            return self._abort("servo_feedback_lost")
        if not obs.electrical_ok:
            return self._abort("electrical_safety_gate")
        if self.state != CycleState.WAIT_START and not obs.scene_clear:
            return self._abort("scene_not_clear")

        ready = {
            CycleState.WAIT_START: obs.start_stable,
            CycleState.SCENE_CLEAR: obs.scene_clear,
            CycleState.FORWARD_PUSH: obs.forward_finished,
            CycleState.FORWARD_ASSESS: obs.outcome_available,
            CycleState.RETRACT: obs.retract_finished,
            CycleState.MOVE_BEHIND: obs.behind_clearance_ok,
            CycleState.DESCEND: obs.descend_finished,
            CycleState.BACK_PUSH: obs.reset_stable,
            CycleState.HOME: obs.home_finished,
            CycleState.AUDIT: obs.audit_passed,
        }[self.state]
        if ready:
            self.state = _NEXT[self.state]
        return self.state

