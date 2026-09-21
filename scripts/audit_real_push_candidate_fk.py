"""Pure-offline FK audit for the three provisional real-Push candidates.

This script deliberately imports only the pure trajectory validation module and
the analytic FK.  It never imports a servo, serial-port, or camera module.  A
PASS means that the candidate file was parsed, interpolated, and evaluated
successfully under the stated *provisional* model; it is not evidence that a
trajectory is safe on hardware or that it will establish contact/succeed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    JOINT_NAMES,
    LOCK_INDEX_BY_CONDITION,
    TICKS_PER_DEGREE,
    SafetyEnvelope,
    interpolate_raw_waypoints,
    load_fixed_raw_trajectory,
    load_safety_envelope,
    sha256_file,
    validate_interpolated_events,
)
from robotarm.envs.fk import forward_kinematics  # noqa: E402


DEFAULT_CANDIDATES = (
    ROOT / "data/real_robot/session_20260901/setup/"
    "trajectory_candidates_unvalidated.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "data/real_robot/session_20260901/setup/"
    "trajectory_candidates_offline_fk_audit.json"
)
DEFAULT_SAFETY = ROOT / "hardware/safety_limits.yaml"
REQUIRED_CONDITIONS = ("intact", "D2", "D3")
AUTHORIZATION = "LOW_SPEED_PILOT_CANDIDATE_NOT_FORMAL_TRAJECTORY"
MODEL_STATUS = "provisional"
MONOTONIC_TOLERANCE_M = 1e-12


def _discover_candidates(path: Path) -> tuple[tuple[str, str], ...]:
    """Return unique (trajectory_id, condition) pairs in file order."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"trajectory_id", "condition"} - fields
        if missing:
            raise ValueError(f"candidate CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    discovered: list[tuple[str, str]] = []
    condition_by_id: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        trajectory_id = row.get("trajectory_id", "").strip()
        condition = row.get("condition", "").strip()
        if not trajectory_id or not condition:
            raise ValueError(
                f"row {row_number}: trajectory_id and condition must be non-empty"
            )
        previous = condition_by_id.setdefault(trajectory_id, condition)
        if previous != condition:
            raise ValueError(
                f"trajectory {trajectory_id!r} occurs under multiple conditions"
            )
        pair = (trajectory_id, condition)
        if pair not in discovered:
            discovered.append(pair)
    if not discovered:
        raise ValueError("candidate CSV contains no trajectories")
    return tuple(discovered)


def _raw_to_radians(
    raw: Sequence[int], safety: SafetyEnvelope,
) -> np.ndarray:
    return np.asarray(
        [
            math.radians(
                joint.direction * (int(value) - joint.zero_raw) / TICKS_PER_DEGREE
            )
            for value, joint in zip(raw, safety.joints)
        ],
        dtype=np.float64,
    )


def _per_joint_event_speed_deg_s(events: Sequence[Any]) -> dict[str, float]:
    maxima = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for previous, current in zip(events, events[1:]):
        dt = float(current.time_s - previous.time_s)
        if dt <= 0.0:
            continue
        delta = np.abs(
            np.asarray(current.targets_raw, dtype=np.float64)
            - np.asarray(previous.targets_raw, dtype=np.float64)
        )
        maxima = np.maximum(maxima, delta / TICKS_PER_DEGREE / dt)
    return {name: float(value) for name, value in zip(JOINT_NAMES, maxima)}


def _audit_one(
    *,
    path: Path,
    trajectory_id: str,
    condition: str,
    safety: SafetyEnvelope,
    maximum_speed_deg_s: float,
) -> dict[str, Any]:
    trajectory = load_fixed_raw_trajectory(
        path,
        trajectory_id=trajectory_id,
        condition=condition,
        safety=safety,
        maximum_speed_deg_s=maximum_speed_deg_s,
    )
    events = interpolate_raw_waypoints(trajectory)
    maximum_event_speed = validate_interpolated_events(
        events, safety, maximum_speed_deg_s
    )
    raw = np.asarray([event.targets_raw for event in events], dtype=np.int64)
    q = np.stack([_raw_to_radians(row, safety) for row in raw])
    tcp = np.stack([forward_kinematics(row) for row in q])
    delta_tcp = tcp[-1] - tcp[0]
    step_dx = np.diff(tcp[:, 0])
    backward = step_dx < -MONOTONIC_TOLERANCE_M

    joints: dict[str, dict[str, Any]] = {}
    for index, joint in enumerate(safety.joints):
        observed_min = int(raw[:, index].min())
        observed_max = int(raw[:, index].max())
        joints[joint.name] = {
            "observed_raw_min": observed_min,
            "observed_raw_max": observed_max,
            "allowed_raw_min": joint.min_raw,
            "allowed_raw_max": joint.max_raw,
            "within_measured_raw_limits": bool(
                observed_min >= joint.min_raw and observed_max <= joint.max_raw
            ),
            "observed_angle_rad_min": float(q[:, index].min()),
            "observed_angle_rad_max": float(q[:, index].max()),
            "maximum_interpolated_speed_deg_s": _per_joint_event_speed_deg_s(events)[
                joint.name
            ],
            "allowed_speed_deg_s": float(
                min(maximum_speed_deg_s, joint.max_speed_deg_s)
            ),
        }

    lock_index = LOCK_INDEX_BY_CONDITION.get(condition)
    if lock_index is None:
        lock = {
            "applicable": False,
            "joint": None,
            "command_exactly_constant_for_all_events": None,
            "raw_range_ticks": None,
            "locked_target_raw": None,
        }
    else:
        locked_raw = raw[:, lock_index]
        lock = {
            "applicable": True,
            "joint": JOINT_NAMES[lock_index],
            "command_exactly_constant_for_all_events": bool(
                np.all(locked_raw == locked_raw[0])
            ),
            "raw_range_ticks": int(np.ptp(locked_raw)),
            "locked_target_raw": int(locked_raw[0]),
        }

    return {
        "condition": condition,
        "parser_status": "PASS",
        "waypoint_count": len(trajectory.waypoints),
        "interpolated_event_count": len(events),
        "duration_s": float(trajectory.duration_s),
        "maximum_waypoint_speed_deg_s": float(
            trajectory.maximum_commanded_speed_deg_s
        ),
        "maximum_interpolated_speed_deg_s": float(maximum_event_speed),
        "speed_limit_deg_s": float(maximum_speed_deg_s),
        "all_events_within_joint_limits": bool(
            all(item["within_measured_raw_limits"] for item in joints.values())
        ),
        "joint_audit": joints,
        "lock_axis_audit": lock,
        "tcp": {
            "coordinate_order": ["x_forward", "y_lateral", "z_vertical"],
            "start_m": [float(value) for value in tcp[0]],
            "end_m": [float(value) for value in tcp[-1]],
            "delta_m": [float(value) for value in delta_tcp],
            "start_to_end_dx_m": float(delta_tcp[0]),
            "lateral_y_range_m": float(np.ptp(tcp[:, 1])),
            "vertical_z_range_m": float(np.ptp(tcp[:, 2])),
            "x_range_m": float(np.ptp(tcp[:, 0])),
            "x_monotonic_nondecreasing": bool(not np.any(backward)),
            "x_monotonic_tolerance_m": MONOTONIC_TOLERANCE_M,
            "backward_x_event_count": int(np.count_nonzero(backward)),
            "minimum_event_dx_m": float(step_dx.min()) if step_dx.size else 0.0,
            "maximum_event_dx_m": float(step_dx.max()) if step_dx.size else 0.0,
        },
    }


def audit_candidate_file(
    candidate_path: Path,
    safety_path: Path = DEFAULT_SAFETY,
    *,
    maximum_speed_deg_s: float = 5.0,
) -> dict[str, Any]:
    """Audit all three candidates without touching any hardware resource."""
    candidate_path = candidate_path.resolve()
    safety_path = safety_path.resolve()
    safety = load_safety_envelope(safety_path)
    pairs = _discover_candidates(candidate_path)
    errors: list[str] = []
    by_condition: dict[str, list[str]] = {}
    for trajectory_id, condition in pairs:
        by_condition.setdefault(condition, []).append(trajectory_id)
    if len(pairs) != 3:
        errors.append(f"expected exactly three candidates, found {len(pairs)}")
    for condition in REQUIRED_CONDITIONS:
        count = len(by_condition.get(condition, []))
        if count != 1:
            errors.append(
                f"expected exactly one {condition} candidate, found {count}"
            )
    unexpected = sorted(set(by_condition) - set(REQUIRED_CONDITIONS))
    if unexpected:
        errors.append(f"unexpected conditions: {unexpected}")

    trajectories: dict[str, Any] = {}
    for trajectory_id, condition in pairs:
        try:
            trajectories[trajectory_id] = _audit_one(
                path=candidate_path,
                trajectory_id=trajectory_id,
                condition=condition,
                safety=safety,
                maximum_speed_deg_s=maximum_speed_deg_s,
            )
        except (OSError, ValueError, FloatingPointError) as exc:
            errors.append(f"{trajectory_id}: {exc}")
            trajectories[trajectory_id] = {
                "condition": condition,
                "parser_status": "FAIL",
                "error": str(exc),
            }

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "PURE_OFFLINE_INTERPOLATED_FK_CANDIDATE_AUDIT",
        "status": "PASS" if not errors else "FAIL",
        "model_status": MODEL_STATUS,
        "authorization": AUTHORIZATION,
        "candidate_file": str(candidate_path),
        "candidate_file_sha256": sha256_file(candidate_path),
        "safety_file": str(safety_path),
        "safety_file_sha256": sha256_file(safety_path),
        "maximum_requested_speed_deg_s": float(maximum_speed_deg_s),
        "trajectory_count": len(pairs),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "trajectories": trajectories,
        "errors": errors,
        "interpretation": {
            "pass_means": (
                "The candidate file is parseable by the formal executor's pure parser, "
                "its integer interpolation satisfies the configured command bounds, and "
                "TCP geometry was computed under the provisional analytic FK."
            ),
            "does_not_establish": [
                "real-robot safety",
                "collision clearance",
                "camera visibility",
                "contact establishment",
                "Push task success",
                "formal trajectory authorization",
            ],
            "hardware_resources_accessed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--maximum-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = audit_candidate_file(
        args.candidates,
        args.safety,
        maximum_speed_deg_s=args.maximum_speed_deg_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
