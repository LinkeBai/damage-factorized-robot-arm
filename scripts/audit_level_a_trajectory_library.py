"""Audit fixed position trajectories before Level-A hardware collection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("j1", "j2", "j3", "j4", "j5")
REQUIRED = {"trajectory_id", "condition", "waypoint_index", "time_s", *JOINTS}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(library: Path, schedule: Path,
          safety_path: Path = ROOT / "hardware/safety_limits.yaml") -> dict:
    errors: list[str] = []
    safety = yaml.safe_load(safety_path.read_text(encoding="utf-8"))
    limits = {item["name"]: item for item in safety["joints"]}
    maximum_speed = min(float(item["max_speed_deg_s"]) for item in safety["joints"])
    with library.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if missing := REQUIRED - set(reader.fieldnames or []):
            return {"status": "FAIL", "errors": [f"library missing columns: {sorted(missing)}"]}
        rows = list(reader)
    with schedule.open(newline="", encoding="utf-8-sig") as handle:
        schedule_rows = list(csv.DictReader(handle))
    scheduled = {row.get("trajectory_id", "") for row in schedule_rows}
    scheduled.discard("")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    if set(grouped) != scheduled:
        errors.append(
            f"trajectory IDs differ: library={sorted(grouped)}, schedule={sorted(scheduled)}")

    summaries = {}
    for trajectory_id, selected in grouped.items():
        conditions = {row["condition"] for row in selected}
        if len(conditions) != 1:
            errors.append(f"{trajectory_id}: condition must be constant")
            continue
        condition = next(iter(conditions))
        expected_condition = {row["condition"] for row in schedule_rows
                              if row.get("trajectory_id") == trajectory_id}
        if expected_condition != {condition}:
            errors.append(f"{trajectory_id}: library/schedule condition mismatch")
        try:
            ordered = sorted(selected, key=lambda row: int(row["waypoint_index"]))
            indices = [int(row["waypoint_index"]) for row in ordered]
            times = np.asarray([float(row["time_s"]) for row in ordered])
            positions = np.asarray([[float(row[name]) for name in JOINTS]
                                    for row in ordered])
        except ValueError:
            errors.append(f"{trajectory_id}: nonnumeric waypoint, time, or joint value")
            continue
        if len(ordered) < 2:
            errors.append(f"{trajectory_id}: requires at least two waypoints")
        if indices != list(range(len(indices))):
            errors.append(f"{trajectory_id}: waypoint indices must start at 0 and be contiguous")
        if len(times) and abs(times[0]) > 1e-12:
            errors.append(f"{trajectory_id}: first waypoint time must be 0")
        dt = np.diff(times)
        if len(dt) and np.any(dt <= 0):
            errors.append(f"{trajectory_id}: waypoint times must be strictly increasing")
        for joint_index, name in enumerate(JOINTS):
            lower = np.radians(float(limits[name]["min_deg"]))
            upper = np.radians(float(limits[name]["max_deg"]))
            if np.any(positions[:, joint_index] < lower) or np.any(positions[:, joint_index] > upper):
                errors.append(f"{trajectory_id}: {name} exceeds measured joint limits")
        speed = (np.abs(np.diff(positions, axis=0)) / dt[:, None]
                 if len(dt) and np.all(dt > 0) else np.empty((0, len(JOINTS))))
        maximum_observed_deg_s = float(np.degrees(speed.max())) if speed.size else 0.0
        if maximum_observed_deg_s > maximum_speed + 1e-9:
            errors.append(
                f"{trajectory_id}: speed {maximum_observed_deg_s:.6g} deg/s exceeds {maximum_speed:g}")
        lock_index = {"D2": 1, "D3": 2}.get(condition)
        if lock_index is not None and np.ptp(positions[:, lock_index]) > 1e-10:
            errors.append(f"{trajectory_id}: locked {JOINTS[lock_index]} command is not constant")
        summaries[trajectory_id] = {
            "condition": condition, "waypoints": len(ordered),
            "duration_s": float(times[-1]) if len(times) else None,
            "maximum_commanded_speed_deg_s": maximum_observed_deg_s,
        }
    return {
        "status": "PASS" if not errors else "FAIL",
        "library": str(library), "library_sha256": digest(library),
        "schedule": str(schedule), "schedule_sha256": digest(schedule),
        "maximum_allowed_speed_deg_s": maximum_speed,
        "trajectories": summaries, "errors": errors,
        "authorization": "TRAJECTORY_LIBRARY_SAFE_TO_FREEZE" if not errors
                         else "DO_NOT_EXECUTE_TRAJECTORY_LIBRARY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/real_robot/trajectory-library-audit.json"))
    args = parser.parse_args()
    payload = audit(args.library, args.schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
