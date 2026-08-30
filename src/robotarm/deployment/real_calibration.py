"""Pure helpers for the ST3215 BT-DPWM calibration protocol."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TICKS_PER_RADIAN = 4096.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class CalibrationPlan:
    topology: str
    locked_index: int
    transitions: int
    frequency_hz: float
    action_amplitude: float
    seed: int


def ticks_to_radians(raw, zero, direction):
    return direction * (np.asarray(raw, dtype=float)-np.asarray(zero, dtype=float)) / TICKS_PER_RADIAN


def radians_to_ticks(angle, zero, direction):
    raw = np.asarray(zero, dtype=float) + direction*np.asarray(angle)*TICKS_PER_RADIAN
    return np.rint(raw).astype(int)


def safe_excitation(plan: CalibrationPlan) -> np.ndarray:
    """Deterministic bounded multisine with a strictly zero locked coordinate."""
    rng = np.random.default_rng(plan.seed)
    time = np.arange(plan.transitions, dtype=float) / plan.frequency_hz
    actions = np.empty((plan.transitions, 5), dtype=float)
    for joint in range(5):
        phase = rng.uniform(-math.pi, math.pi)
        frequency = 0.17 + 0.06*joint
        actions[:, joint] = plan.action_amplitude*np.sin(2*math.pi*frequency*time+phase)
    actions[:, plan.locked_index] = 0.0
    return np.clip(actions, -plan.action_amplitude, plan.action_amplitude)


def build_state(position, velocity, object_pose) -> np.ndarray:
    state = np.concatenate((np.asarray(position, float), np.asarray(velocity, float),
                            np.asarray(object_pose, float)))
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("real calibration state must contain 14 finite values")
    return state


def validate_transition_arrays(states, actions, locked_index, lock_angle,
                               maximum_lock_drift_rad):
    states, actions = np.asarray(states), np.asarray(actions)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError("states must have shape [K+1, 14]")
    if actions.shape != (states.shape[0]-1, 5):
        raise ValueError("actions must have shape [K, 5]")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("calibration arrays contain non-finite values")
    if not np.allclose(actions[:, locked_index], 0.0, atol=1e-12):
        raise ValueError("locked-joint actions must be exactly zero")
    drift = np.max(np.abs(states[:, locked_index]-lock_angle))
    if drift > maximum_lock_drift_rad:
        raise ValueError(f"locked joint drift {drift:.6f} exceeds safety limit")
    return {"transitions": len(actions), "maximum_lock_drift_rad": float(drift)}
