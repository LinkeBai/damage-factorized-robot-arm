"""Build a read-only strict-goal ledger for every recorded real Push trial.

The script never edits source manifests, videos, or older analyses.  It applies
one radial endpoint gate to every trial that already has a confidence-gated
image-space endpoint assessment and records missing evidence explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_summary(trial: Path) -> tuple[Path | None, list[Path]]:
    candidates = []
    for path in trial.rglob("yellow_cube_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        image_task = payload.get("image_task")
        if image_task and image_task.get("goal_center_px") is not None:
            candidates.append(path)
    candidates.sort(key=lambda path: (
        "strict3px" not in path.as_posix().lower(),
        "aligned" not in path.as_posix().lower(),
        path.as_posix(),
    ))
    return (candidates[0] if candidates else None), candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--tolerance-px", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_px) or args.tolerance_px <= 0:
        parser.error("--tolerance-px must be finite and positive")

    rows = []
    for group in ("confirmatory_trials", "pilot_trials"):
        group_dir = args.session / group
        if not group_dir.is_dir():
            continue
        for trial in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            videos = sorted(trial.glob("daheng_*_raw.avi"))
            manifest_path = trial / "run_manifest.json"
            if not videos or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected, candidates = choose_summary(trial)
            row = {
                "group": group,
                "trial_id": manifest.get("trial_id", trial.name),
                "condition": manifest.get("condition"),
                "trajectory_id": manifest.get("trajectory_id"),
                "manifest_status": manifest.get("status"),
                "acquisition_valid": manifest.get("status") == "ACQUISITION_COMPLETE_UNASSESSED",
                "strict_gate_type": "radial_endpoint_error_px",
                "strict_tolerance_px": args.tolerance_px,
                "assessment_status": "UNASSESSED_NO_FROZEN_GOAL_TRACKING",
                "strict_success": None,
                "goal_x_px": None,
                "goal_y_px": None,
                "endpoint_error_x_px": None,
                "endpoint_error_y_px": None,
                "endpoint_error_radial_px": None,
                "tracking_confidence_pass": None,
                "selected_tracking_summary": None,
                "selected_tracking_sha256": None,
                "candidate_tracking_summary_count": len(candidates),
            }
            if selected is not None:
                payload = json.loads(selected.read_text(encoding="utf-8"))
                image_task = payload["image_task"]
                goal = image_task.get("goal_center_px")
                error_xy = image_task.get("endpoint_error_xy_px")
                radial = image_task.get("endpoint_error_radial_px")
                confidence = bool(payload.get("confidence_gate", {}).get("pass"))
                assessed = bool(image_task.get("assessed") and confidence and radial is not None)
                strict_success = bool(float(radial) <= args.tolerance_px) if assessed else None
                row.update({
                    "assessment_status": (
                        "ASSESSED_VALID_ACQUISITION" if row["acquisition_valid"] and assessed
                        else "ASSESSED_INVALID_ACQUISITION" if assessed
                        else "UNASSESSED_TRACKING_CONFIDENCE_FAILED"
                    ),
                    "strict_success": strict_success,
                    "goal_x_px": goal[0] if goal else None,
                    "goal_y_px": goal[1] if goal else None,
                    "endpoint_error_x_px": error_xy[0] if error_xy else None,
                    "endpoint_error_y_px": error_xy[1] if error_xy else None,
                    "endpoint_error_radial_px": radial,
                    "tracking_confidence_pass": confidence,
                    "selected_tracking_summary": str(selected.resolve()),
                    "selected_tracking_sha256": sha256_file(selected),
                })
            rows.append(row)

    valid_assessed = [row for row in rows
                      if row["assessment_status"] == "ASSESSED_VALID_ACQUISITION"]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "success_iff": (
                "confidence-gated robust endpoint radial error <= "
                f"{args.tolerance_px:g} px"
            ),
            "tolerance_px": args.tolerance_px,
            "source_artifacts_modified": False,
            "setup_and_recovery_runs_excluded": True,
            "aborted_acquisitions_never_count_as_valid_trials": True,
        },
        "counts": {
            "recorded_trial_packets": len(rows),
            "valid_assessed_trials": len(valid_assessed),
            "valid_strict_successes": sum(row["strict_success"] is True for row in valid_assessed),
            "valid_strict_failures": sum(row["strict_success"] is False for row in valid_assessed),
            "invalid_or_unassessed": len(rows) - len(valid_assessed),
        },
        "trials": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["trial_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
