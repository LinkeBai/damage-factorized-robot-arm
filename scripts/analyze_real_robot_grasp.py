"""Validate and summarize fixed-pregrasp short-lift feasibility trials."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED = {
    "trial_id", "condition", "method", "position_id", "trial_order", "aborted",
    "max_lock_error_rad", "pregrasp_reached", "gripper_closed", "lift_height_m",
    "retained_after_3s", "success", "camera_left_video",
    "camera_horizontal_video", "control_log", "failure_code",
}
BOOL_FIELDS = ("pregrasp_reached", "gripper_closed", "retained_after_3s", "success")


def flag(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Invalid Boolean value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/real_robot/grasp-feasibility-summary.json"),
    )
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Trial table contains no rows")
    if len({row["trial_id"] for row in rows}) != len(rows):
        raise SystemExit("trial_id must be unique")
    orders = [row["trial_order"].strip() for row in rows]
    if any(not value for value in orders) or len(set(orders)) != len(orders):
        raise SystemExit("trial_order must be present and unique")

    errors: list[str] = []
    valid: list[dict[str, str]] = []
    for line, row in enumerate(rows, start=2):
        try:
            aborted = flag(row["aborted"])
        except ValueError as exc:
            errors.append(f"line {line}: {exc}")
            continue
        if aborted:
            if not row["failure_code"].strip():
                errors.append(f"line {line}: aborted trial requires failure_code")
            continue
        required_values = set(BOOL_FIELDS) | {
            "max_lock_error_rad", "lift_height_m", "camera_left_video",
            "camera_horizontal_video", "control_log",
        }
        missing_values = sorted(name for name in required_values if not row[name].strip())
        if missing_values:
            errors.append(f"line {line}: non-aborted trial missing {missing_values}")
            continue
        for name in BOOL_FIELDS:
            try:
                flag(row[name])
            except ValueError as exc:
                errors.append(f"line {line}: {exc}")
        for name in ("max_lock_error_rad", "lift_height_m"):
            try:
                value = float(row[name])
                if not np.isfinite(value) or value < 0:
                    raise ValueError
            except ValueError:
                errors.append(f"line {line}: {name} must be finite and non-negative")
        if args.require_files:
            for name in ("camera_left_video", "camera_horizontal_video", "control_log"):
                if not Path(row[name]).is_file():
                    errors.append(f"line {line}: {name} does not exist: {row[name]}")
        valid.append(row)
    if errors:
        raise SystemExit("Trial validity gate failed:\n" + "\n".join(errors[:20]))

    conditions: dict[str, dict] = {}
    for condition in sorted({row["condition"] for row in valid}):
        selected = [row for row in valid if row["condition"] == condition]
        conditions[condition] = {
            "trials": len(selected),
            "pregrasp_reach_rate": float(np.mean([
                flag(row["pregrasp_reached"]) for row in selected
            ])),
            "closure_rate": float(np.mean([
                flag(row["gripper_closed"]) for row in selected
            ])),
            "retention_rate_3s": float(np.mean([
                flag(row["retained_after_3s"]) for row in selected
            ])),
            "success_rate": float(np.mean([flag(row["success"]) for row in selected])),
            "mean_lift_height_m": float(np.mean([
                float(row["lift_height_m"]) for row in selected
            ])),
            "max_lock_error_rad": float(max(
                float(row["max_lock_error_rad"]) for row in selected
            )),
        }
    payload = {
        "source": str(args.csv),
        "scope": "fixed-pregrasp short-lift feasibility only; no learned-grasp claim",
        "rows": len(rows),
        "valid_rows": len(valid),
        "aborted_rows": len(rows) - len(valid),
        "conditions": conditions,
        "failure_codes": dict(Counter(row["failure_code"] or "none" for row in rows)),
        "all_required_files_checked": args.require_files,
        "claim_level": "feasibility" if valid else "no evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
