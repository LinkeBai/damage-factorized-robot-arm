"""Globally probe level Cartesian reach after a frozen joint lock.

This is an offline diagnostic for the original five-axis arm model.  It never
imports a camera or servo driver and never authorizes hardware motion.
"""
from __future__ import annotations

import argparse
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


START_RAW = np.asarray([2085, 2635, 2603, 2740, 2077], dtype=np.int64)


def raw_to_q(raw: np.ndarray, safety) -> np.ndarray:
    return np.asarray([
        math.radians(joint.direction * (int(value) - joint.zero_raw) / TICKS_PER_DEGREE)
        for value, joint in zip(raw, safety.joints)
    ])


def q_to_raw(q: np.ndarray, safety) -> list[int]:
    return [
        int(round(joint.zero_raw + joint.direction * math.degrees(float(value)) * TICKS_PER_DEGREE))
        for value, joint in zip(q, safety.joints)
    ]


def solve_target(target, *, locked_index, locked_value, lower, upper, seeds):
    free = [index for index in range(5) if index != locked_index]
    best = None
    for seed in seeds:
        x0 = np.clip(np.asarray(seed)[free], lower[free], upper[free])

        def residual(values):
            q = np.empty(5, dtype=np.float64)
            q[free] = values
            q[locked_index] = locked_value
            return forward_kinematics(q) - target

        result = least_squares(
            residual,
            x0,
            bounds=(lower[free], upper[free]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=3000,
        )
        q = np.empty(5, dtype=np.float64)
        q[free] = result.x
        q[locked_index] = locked_value
        error = float(np.linalg.norm(residual(result.x)))
        if best is None or error < best[0]:
            best = (error, q, result)
    assert best is not None
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("D2", "D3", "D4"), required=True)
    parser.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    parser.add_argument("--min-dx-m", type=float, default=0.0)
    parser.add_argument("--max-dx-m", type=float, default=0.18)
    parser.add_argument("--step-m", type=float, default=0.005)
    parser.add_argument("--random-seeds", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    safety = load_safety_envelope(args.safety.resolve())
    q_start = raw_to_q(START_RAW, safety)
    start_xyz = forward_kinematics(q_start)
    lower = np.radians([joint.min_deg for joint in safety.joints])
    upper = np.radians([joint.max_deg for joint in safety.joints])
    locked_index = LOCK_INDEX_BY_CONDITION[args.condition]
    locked_value = float(q_start[locked_index])
    rng = np.random.default_rng(20260903)
    random_seeds = rng.uniform(lower, upper, size=(args.random_seeds, 5))
    random_seeds[:, locked_index] = locked_value
    continuation = q_start.copy()
    rows = []

    if args.min_dx_m > args.max_dx_m:
        raise ValueError("min-dx-m must not exceed max-dx-m")
    for dx in np.arange(args.min_dx_m, args.max_dx_m + args.step_m / 2, args.step_m):
        target = start_xyz + np.asarray([dx, 0.0, 0.0])
        seeds = np.vstack((q_start, continuation, random_seeds))
        error, q, result = solve_target(
            target,
            locked_index=locked_index,
            locked_value=locked_value,
            lower=lower,
            upper=upper,
            seeds=seeds,
        )
        continuation = q
        actual = forward_kinematics(q)
        rows.append({
            "requested_dx_m": float(dx),
            "residual_m": error,
            "actual_delta_m": [float(value) for value in actual - start_xyz],
            "solution_q_rad": [float(value) for value in q],
            "solution_raw_rounded": q_to_raw(q, safety),
            "active_lower_bound_joints": [
                safety.joints[i].name for i in range(5) if abs(q[i] - lower[i]) < 1e-5
            ],
            "active_upper_bound_joints": [
                safety.joints[i].name for i in range(5) if abs(q[i] - upper[i]) < 1e-5
            ],
            "solver_success": bool(result.success),
        })

    def largest_at(tolerance):
        feasible = [row for row in rows if row["residual_m"] <= tolerance]
        return None if not feasible else max(row["requested_dx_m"] for row in feasible)

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "offline multi-start bounded least-squares reachability probe",
        "model": "original five-axis arm analytic FK; provisional TCP",
        "hardware_accessed": False,
        "condition": args.condition,
        "locked_joint": safety.joints[locked_index].name,
        "locked_start_raw": int(START_RAW[locked_index]),
        "start_raw": START_RAW.tolist(),
        "start_xyz_m": [float(value) for value in start_xyz],
        "fixed_target_plane_yz_m": [float(start_xyz[1]), float(start_xyz[2])],
        "maximum_requested_dx_m": args.max_dx_m,
        "minimum_requested_dx_m": args.min_dx_m,
        "step_m": args.step_m,
        "random_seed_count_per_target": args.random_seeds,
        "largest_dx_with_residual_lte_1mm": largest_at(0.001),
        "largest_dx_with_residual_lte_5mm": largest_at(0.005),
        "rows": rows,
        "claim_boundary": (
            "A failed target is evidence under this bounded analytic model and search, "
            "not a formal proof of physical unreachability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "condition", "locked_joint", "largest_dx_with_residual_lte_1mm",
        "largest_dx_with_residual_lte_5mm", "claim_boundary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
