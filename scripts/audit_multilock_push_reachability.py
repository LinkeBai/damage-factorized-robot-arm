"""Screen every non-empty joint-lock subset for task-axis reachability.

Pure offline screening: PASS is necessary but not sufficient for hardware use.
It does not authorize motion or test collision, load, contact, or lock drift.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotarm.deployment.fixed_raw_trajectory import TICKS_PER_DEGREE, load_safety_envelope
from robotarm.envs.fk import forward_kinematics

START_RAW = np.asarray([2085, 2635, 2603, 2740, 2077], dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-distance-m", type=float, default=0.0526383)
    p.add_argument("--random-seeds", type=int, default=24)
    p.add_argument("--tolerance-m", type=float, default=.005)
    p.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    p.add_argument(
        "--probe-root", type=Path,
        default=ROOT / "results/real_robot/multilock-safety",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    safety = load_safety_envelope(args.safety)
    authorized_multilocks: set[str] = set()
    if args.probe_root.is_dir():
        for manifest_path in args.probe_root.glob("*/manifest.json"):
            try:
                probe = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            condition = probe.get("condition")
            if probe.get("status") == "PASS" and isinstance(condition, str):
                authorized_multilocks.add(condition)
    q0 = np.asarray([math.radians(j.direction * (r-j.zero_raw) / TICKS_PER_DEGREE) for r,j in zip(START_RAW, safety.joints)])
    lo = np.radians([j.min_deg for j in safety.joints]); hi = np.radians([j.max_deg for j in safety.joints])
    start = forward_kinematics(q0); target = start + np.asarray([args.target_distance_m, 0, 0])
    rng = np.random.default_rng(20260903)
    rows = []
    for count in range(1, 6):
        for locked in itertools.combinations(range(5), count):
            free = [i for i in range(5) if i not in locked]
            best = float(np.linalg.norm(start-target)); best_q = q0.copy()
            if free:
                seeds = np.vstack([q0, rng.uniform(lo, hi, size=(args.random_seeds, 5))])
                for seed in seeds:
                    def residual(x):
                        q=q0.copy(); q[free]=x
                        return forward_kinematics(q)-target
                    result=least_squares(residual, np.clip(seed[free],lo[free],hi[free]), bounds=(lo[free],hi[free]), max_nfev=1000)
                    err=float(np.linalg.norm(residual(result.x)))
                    if err < best:
                        best=err; best_q=q0.copy(); best_q[free]=result.x
            names=[f"J{i+1}" for i in locked]
            condition = "+".join(names)
            # Every single axis has a measured probe.  A multi-lock subset is
            # authorized only by its own retained PASS manifest.
            hardware_lock_authorized = count == 1 or condition in authorized_multilocks
            rows.append({"locked_joints":names,"lock_count":count,"residual_m":best,
                         "reachability_pass":best<=args.tolerance_m,
                         "hardware_lock_authorized_currently":hardware_lock_authorized,
                         "solution_q_rad":best_q.tolist(),
                         "hardware_action":"ELIGIBLE_FOR_COLLISION_LOAD_PREFLIGHT" if best<=args.tolerance_m and hardware_lock_authorized else "DO_NOT_EXECUTE"})
    payload={"schema_version":1,"generated_utc":datetime.now(timezone.utc).isoformat(),
             "hardware_accessed":False,"target_distance_m":args.target_distance_m,
             "tolerance_m":args.tolerance_m,
             "authorized_multilocks": sorted(authorized_multilocks), "rows":rows,
             "claim_boundary":"Reachability PASS is not hardware authorization; collision, load, contact, speed, and lock-drift gates remain mandatory."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"subsets":len(rows),"reachable":sum(r["reachability_pass"] for r in rows),
                      "currently_authorized_and_reachable":sum(r["hardware_action"].startswith("ELIGIBLE") for r in rows)},indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
