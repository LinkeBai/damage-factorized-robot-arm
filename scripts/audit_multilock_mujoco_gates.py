"""Offline MuJoCo path gates for every row in the 31-condition reachability audit.

This checks the straight joint-space path from the frozen real-arm start to the
bounded IK endpoint.  It is a conservative candidate-path screen, not proof
that another collision-free path does or does not exist and not hardware
authorization.  Contacts involving the task block are ignored; any arm-table
or arm-arm collision-proxy contact is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import yaml

TICKS_PER_DEG = 4096.0 / 360.0
START_RAW = np.asarray([2085, 2635, 2603, 2740, 2077], dtype=float)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")

    source = json.loads(args.reachability.read_text(encoding="utf-8"))
    safety_doc = yaml.safe_load(args.safety.read_text(encoding="utf-8"))
    joints = safety_doc["joints"]
    q0 = np.asarray([
        math.radians(j["direction"] * (raw - j["zero_raw"]) / TICKS_PER_DEG)
        for raw, j in zip(START_RAW, joints)
    ])
    limits = np.radians(np.asarray([[j["min_deg"], j["max_deg"]] for j in joints]))
    speed_limit = min(float(j["max_speed_deg_s"]) for j in joints)
    max_hold = float(safety_doc["damage_test"]["max_lock_hold_s"])
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    rows = []
    for source_row in source["rows"]:
        q1 = np.asarray(source_row["solution_q_rad"], dtype=float)
        within_limits = bool(np.all(q1 >= limits[:, 0] - 1e-9) and np.all(q1 <= limits[:, 1] + 1e-9))
        minimum_duration = float(np.max(np.abs(np.degrees(q1 - q0))) / speed_limit)
        collision_samples = []
        for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, args.samples)):
            data.qpos[:5] = q0 + alpha * (q1 - q0)
            mujoco.mj_forward(model, data)
            contacts = []
            for contact in data.contact[:data.ncon]:
                first = model.geom(int(contact.geom1)).name
                second = model.geom(int(contact.geom2)).name
                if "block_geom" in (first, second):
                    continue
                contacts.append({"geom1": first, "geom2": second, "distance_m": float(contact.dist)})
            if contacts:
                collision_samples.append({"sample_index": sample_index, "alpha": float(alpha), "contacts": contacts})
        rows.append({
            "locked_joints": source_row["locked_joints"],
            "reachability_pass": source_row["reachability_pass"],
            "joint_limit_gate": "PASS" if within_limits else "FAIL",
            "linear_path_collision_gate": "PASS" if not collision_samples else "FAIL",
            "collision_sample_count": len(collision_samples),
            "first_collision": collision_samples[0] if collision_samples else None,
            "minimum_linear_duration_s_at_speed_limit": minimum_duration,
            "speed_within_lock_hold_gate": "PASS" if minimum_duration <= max_hold else "FAIL_CANDIDATE_PATH_ONLY",
            "locked_axes_constant_by_construction": True,
        })
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_accessed": False,
        "frozen_start_raw": START_RAW.astype(int).tolist(),
        "samples_per_path": args.samples,
        "model": str(args.model.resolve()),
        "model_sha256": digest(args.model),
        "safety_sha256": digest(args.safety),
        "rows": rows,
        "claim_boundary": (
            "PASS/FAIL applies only to the straight joint-space path to the stored bounded-IK endpoint; "
            "it is neither a hardware safety authorization nor a proof about all possible paths."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "limit_pass": sum(row["joint_limit_gate"] == "PASS" for row in rows),
        "linear_collision_pass": sum(row["linear_path_collision_gate"] == "PASS" for row in rows),
        "speed_hold_pass": sum(row["speed_within_lock_hold_gate"] == "PASS" for row in rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
