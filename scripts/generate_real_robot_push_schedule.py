"""Generate a hashed randomized-block schedule before real Push collection."""
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
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fault-pairs", type=int, default=10)
    parser.add_argument("--intact-pairs", type=int, default=5)
    parser.add_argument("--conditions", default="D2,D3")
    parser.add_argument("--methods", default="nominal,global_matched")
    parser.add_argument("--positions", default="A,B,C")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fault_pairs < 1 or args.intact_pairs < 0:
        raise ValueError("pair counts must be non-negative and fault-pairs positive")
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    positions = [value.strip() for value in args.positions.split(",") if value.strip()]
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("at least two unique methods are required")
    if not conditions or not positions:
        raise ValueError("conditions and positions must be non-empty")
    rng = np.random.default_rng(args.seed)

    blocks: list[dict] = []
    condition_counts = {"intact": args.intact_pairs, **{
        condition: args.fault_pairs for condition in conditions
    }}
    pair_index = 1
    for condition, count in condition_counts.items():
        for repetition in range(count):
            blocks.append({
                "pair_id": f"P{pair_index:03d}",
                "condition": condition,
                "position_id": positions[repetition % len(positions)],
                "methods": list(rng.permutation(methods)),
            })
            pair_index += 1
    rng.shuffle(blocks)
    rows: list[dict[str, str | int]] = []
    order = 1
    for block in blocks:
        for method in block["methods"]:
            row = {field: "" for field in FIELDS}
            row.update({
                "pair_id": block["pair_id"], "condition": block["condition"],
                "method": str(method), "position_id": block["position_id"],
                "trial_order": order, "aborted": 0,
            })
            rows.append(row)
            order += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "original_5dof_real_push_v1",
        "status": "freeze_before_first_method_trial",
        "seed": args.seed,
        "conditions": condition_counts,
        "methods": methods,
        "positions": positions,
        "blocks": len(blocks),
        "trials": len(rows),
        "output": str(args.output),
        "sha256": sha256(args.output),
        "rule": "block order and within-block method order randomized; all methods share pair and position",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
