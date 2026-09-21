"""Plan a continuous level Push segment inside a locked-joint reachable interval.

The planner searches fixed wrist-yaw slices and follows one continuous IK
branch across Cartesian waypoints. It is offline-only and does not authorize
hardware execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    LOCK_INDEX_BY_CONDITION,
    TICKS_PER_DEGREE,
    load_safety_envelope,
)
from robotarm.envs.fk import forward_kinematics  # noqa: E402


FROZEN_REFERENCE_RAW = np.asarray([2085, 2635, 2603, 2740, 2077], dtype=np.int64)


def raw_to_q(raw, safety):
    return np.asarray([
        math.radians(joint.direction * (int(value) - joint.zero_raw) / TICKS_PER_DEGREE)
        for value, joint in zip(raw, safety.joints)
    ])


def q_to_raw(q, safety):
    return np.asarray([
        int(round(joint.zero_raw + joint.direction * math.degrees(float(value)) * TICKS_PER_DEGREE))
        for value, joint in zip(q, safety.joints)
    ], dtype=np.int64)


def solve_slice(target, *, fixed, lower, upper, seeds):
    free = [index for index in range(5) if index not in fixed]
    candidates = []
    for seed in seeds:
        x0 = np.clip(np.asarray(seed)[free], lower[free], upper[free])

        def residual(values):
            q = np.empty(5)
            q[free] = values
            for index, value in fixed.items():
                q[index] = value
            return forward_kinematics(q) - target

        result = least_squares(
            residual, x0, bounds=(lower[free], upper[free]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000,
        )
        q = np.empty(5)
        q[free] = result.x
        for index, value in fixed.items():
            q[index] = value
        error = float(np.linalg.norm(residual(result.x)))
        if error <= 0.001:
            candidates.append((error, q))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("D2", "D3"), required=True)
    parser.add_argument("--start-dx-m", type=float, required=True)
    parser.add_argument("--end-dx-m", type=float, required=True)
    parser.add_argument("--waypoint-count", type=int, default=7)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--wrist-yaw-deg", type=float, default=None,
        help="fix one wrist-yaw slice; default searches the configured yaw range",
    )
    parser.add_argument(
        "--free-wrist-yaw", action="store_true",
        help="solve a continuous redundant IK path with only the failed joint fixed",
    )
    parser.add_argument("--random-seeds", type=int, default=16)
    parser.add_argument("--maximum-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--maximum-lock-hold-s", type=float, default=10.0)
    parser.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.waypoint_count < 2 or args.end_dx_m <= args.start_dx_m:
        raise ValueError("need at least two waypoints and end-dx-m > start-dx-m")

    safety = load_safety_envelope(args.safety.resolve())
    q_reference = raw_to_q(FROZEN_REFERENCE_RAW, safety)
    reference_xyz = forward_kinematics(q_reference)
    lower = np.radians([joint.min_deg for joint in safety.joints])
    upper = np.radians([joint.max_deg for joint in safety.joints])
    lock_index = LOCK_INDEX_BY_CONDITION[args.condition]
    rng = np.random.default_rng(20260903)
    random_bank = rng.uniform(lower, upper, size=(args.random_seeds, 5))
    random_bank[:, lock_index] = q_reference[lock_index]
    dx_values = np.linspace(args.start_dx_m, args.end_dx_m, args.waypoint_count)
    if args.free_wrist_yaw and args.wrist_yaw_deg is not None:
        raise ValueError("--free-wrist-yaw and --wrist-yaw-deg are mutually exclusive")
    if args.free_wrist_yaw:
        yaw_values = [None]
    elif args.wrist_yaw_deg is None:
        yaw_values = np.radians(np.arange(
            math.degrees(lower[4]), math.degrees(upper[4]) + args.yaw_step_deg / 2,
            args.yaw_step_deg,
        ))
    else:
        requested_yaw = math.radians(args.wrist_yaw_deg)
        if not lower[4] <= requested_yaw <= upper[4]:
            raise ValueError("wrist-yaw-deg is outside the measured joint range")
        yaw_values = np.asarray([requested_yaw])
    plans = []

    for yaw in yaw_values:
        fixed = {lock_index: float(q_reference[lock_index])}
        if yaw is not None:
            fixed[4] = float(yaw)
        path = []
        previous = q_reference.copy()
        feasible = True
        residuals = []
        for dx in dx_values:
            target = reference_xyz + np.asarray([dx, 0.0, 0.0])
            seeds = np.vstack((previous, q_reference, random_bank))
            candidates = solve_slice(
                target, fixed=fixed, lower=lower, upper=upper, seeds=seeds
            )
            if not candidates:
                feasible = False
                break
            _, chosen = min(
                candidates,
                key=lambda item: (
                    float(np.max(np.abs(item[1] - previous))),
                    float(np.linalg.norm(item[1] - previous)),
                    item[0],
                ),
            )
            residuals.append(float(np.linalg.norm(forward_kinematics(chosen) - target)))
            path.append(chosen)
            previous = chosen
        if not feasible:
            continue
        raw_path = np.stack([q_to_raw(q, safety) for q in path])
        delta_deg = np.abs(np.diff(raw_path, axis=0)) / TICKS_PER_DEGREE
        segment_durations = np.max(delta_deg, axis=1) / args.maximum_speed_deg_s
        total_duration = float(np.sum(segment_durations))
        plans.append({
            "wrist_yaw_mode": "variable" if yaw is None else "fixed",
            "wrist_yaw_deg": None if yaw is None else float(math.degrees(yaw)),
            "wrist_yaw_range_deg": [
                float(np.min(np.degrees(np.asarray(path)[:, 4]))),
                float(np.max(np.degrees(np.asarray(path)[:, 4]))),
            ],
            "q_path": path,
            "raw_path": raw_path,
            "residuals_m": residuals,
            "segment_durations_s": segment_durations,
            "total_minimum_duration_s": total_duration,
            "maximum_single_segment_delta_deg": float(np.max(delta_deg)),
        })

    if not plans:
        raise RuntimeError("no continuous fixed-wrist-yaw IK path met the 1 mm waypoint gate")
    best = min(plans, key=lambda plan: plan["total_minimum_duration_s"])
    times = np.concatenate(([0.0], np.cumsum(best["segment_durations_s"])))
    # Add a small scheduling margin while preserving the speed ceiling.
    if times[-1] > 0:
        times *= 1.01

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "condition", "waypoint_index", "time_s", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw"])
        for index, (time_s, raw) in enumerate(zip(times, best["raw_path"])):
            writer.writerow([args.trajectory_id, args.condition, index, f"{time_s:.6f}", *raw.tolist()])

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if times[-1] <= args.maximum_lock_hold_s else "FAIL_LOCK_HOLD_DURATION",
        "hardware_accessed": False,
        "condition": args.condition,
        "locked_joint": safety.joints[lock_index].name,
        "start_dx_m": args.start_dx_m,
        "end_dx_m": args.end_dx_m,
        "net_cartesian_push_m": args.end_dx_m - args.start_dx_m,
        "waypoint_count": args.waypoint_count,
        "selected_wrist_yaw_mode": best["wrist_yaw_mode"],
        "selected_wrist_yaw_deg": best["wrist_yaw_deg"],
        "selected_wrist_yaw_range_deg": best["wrist_yaw_range_deg"],
        "maximum_waypoint_residual_m": max(best["residuals_m"]),
        "minimum_duration_at_speed_limit_s": best["total_minimum_duration_s"],
        "scheduled_duration_s": float(times[-1]),
        "maximum_speed_deg_s": args.maximum_speed_deg_s,
        "maximum_lock_hold_s": args.maximum_lock_hold_s,
        "raw_waypoints": best["raw_path"].tolist(),
        "times_s": times.tolist(),
        "candidate_yaw_slice_count": len(plans),
        "output_csv": str(args.output_csv.resolve()),
        "claim_boundary": "offline provisional-FK candidate; requires independent audit and visual low-speed validation before hardware use",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
