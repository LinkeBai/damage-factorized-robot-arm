"""Generate a randomized Level-A physical-feasibility schedule.

This schedule deliberately has no learned-method labels.  It may be generated
only after one safe fixed trajectory per condition has been manually validated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "pair_id", "condition", "method", "position_id", "trial_order", "aborted",
    "max_lock_error_rad", "reached", "contact", "endpoint_error_m", "success",
    "camera_left_video", "camera_horizontal_video", "control_log", "failure_code",
    "trajectory_id",
]


def build(seed: int, repetitions: int,
          trajectory_ids: dict[str, str]) -> list[dict[str, str]]:
    if repetitions < 10:
        raise ValueError("formal Level-A schedule requires at least 10 trials per condition")
    if set(trajectory_ids) != {"intact", "D2", "D3"}:
        raise ValueError("one validated trajectory ID is required for intact, D2, and D3")
    if any(not value.strip() for value in trajectory_ids.values()):
        raise ValueError("trajectory IDs must be non-empty")
    rows = []
    for condition in ("intact", "D2", "D3"):
        for index in range(repetitions):
            rows.append({
                "pair_id": f"L{condition}_{index + 1:02d}",
                "condition": condition,
                "method": "fixed_safe_trajectory",
                "position_id": ("A", "B", "C")[index % 3],
                "trial_order": "",
                "aborted": "0",
                "max_lock_error_rad": "", "reached": "", "contact": "",
                "endpoint_error_m": "", "success": "",
                "camera_left_video": "", "camera_horizontal_video": "",
                "control_log": "", "failure_code": "",
                "trajectory_id": trajectory_ids[condition],
            })
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    for order, row in enumerate(rows, start=1):
        row["trial_order"] = str(order)
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--intact-trajectory-id", required=True)
    parser.add_argument("--d2-trajectory-id", required=True)
    parser.add_argument("--d3-trajectory-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = {"intact": args.intact_trajectory_id,
           "D2": args.d2_trajectory_id, "D3": args.d3_trajectory_id}
    rows = build(args.seed, args.repetitions, ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "original_5dof_real_push_level_a_v1",
        "claim_boundary": "physical feasibility only; no learned-method comparison",
        "seed": args.seed, "repetitions_per_condition": args.repetitions,
        "trials": len(rows), "trajectory_ids": ids,
        "schedule_sha256": digest(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
