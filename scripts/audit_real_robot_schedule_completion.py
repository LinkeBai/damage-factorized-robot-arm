"""Verify a completed real-robot trial log against its frozen schedule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


IDENTITY_FIELDS = (
    "trial_order", "pair_id", "condition", "method", "position_id", "trajectory_id")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(schedule: Path, completed: Path) -> dict:
    expected, observed = read(schedule), read(completed)
    errors: list[str] = []
    schedule_fields = set(expected[0]) if expected else set()
    completed_fields = set(observed[0]) if observed else set()
    required = set(IDENTITY_FIELDS)
    if missing := required - schedule_fields:
        errors.append(f"frozen schedule missing identity fields: {sorted(missing)}")
    if missing := required - completed_fields:
        errors.append(f"completed log missing identity fields: {sorted(missing)}")
    if len(expected) != len(observed):
        errors.append(
            f"row count differs: frozen={len(expected)}, completed={len(observed)}")

    expected_by_order = {row.get("trial_order", ""): row for row in expected}
    observed_by_order = {row.get("trial_order", ""): row for row in observed}
    if len(expected_by_order) != len(expected):
        errors.append("frozen schedule has duplicate or blank trial_order")
    if len(observed_by_order) != len(observed):
        errors.append("completed log has duplicate or blank trial_order")
    missing_orders = sorted(set(expected_by_order) - set(observed_by_order))
    extra_orders = sorted(set(observed_by_order) - set(expected_by_order))
    if missing_orders:
        errors.append(f"completed log missing trial orders: {missing_orders}")
    if extra_orders:
        errors.append(f"completed log has unexpected trial orders: {extra_orders}")

    changed = []
    for order in sorted(set(expected_by_order) & set(observed_by_order), key=lambda x: int(x)):
        before, after = expected_by_order[order], observed_by_order[order]
        differences = {
            field: {"frozen": before.get(field, ""), "completed": after.get(field, "")}
            for field in IDENTITY_FIELDS if before.get(field, "") != after.get(field, "")
        }
        if differences:
            changed.append({"trial_order": order, "differences": differences})
    if changed:
        errors.append(f"{len(changed)} trial identity rows changed after freeze")

    return {
        "status": "PASS" if not errors else "FAIL",
        "frozen_schedule": str(schedule),
        "completed_log": str(completed),
        "frozen_schedule_sha256": digest(schedule),
        "completed_log_sha256": digest(completed),
        "frozen_rows": len(expected), "completed_rows": len(observed),
        "identity_fields": list(IDENTITY_FIELDS),
        "changed_identity_rows": changed,
        "errors": errors,
        "claim_boundary": (
            "PASS establishes schedule identity preservation only; measurement "
            "validity and raw-file completeness require the Push analyzer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("completed", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/real_robot/schedule-completion-audit.json"))
    args = parser.parse_args()
    payload = audit(args.schedule, args.completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
