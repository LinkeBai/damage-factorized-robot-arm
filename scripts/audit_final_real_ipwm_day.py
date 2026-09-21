"""Audit the frozen final-day real-IPWM schedule without inventing evidence."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


TERMINAL = {"GO", "NO_GO", "INVALID", "PREFLIGHT_EXCLUDED"}
REQUIRED_PUSH = {
    "run_manifest.json",
    "daheng_FDE23080341_raw.avi",
    "directshow_index1_raw.avi",
    "frame_timestamps.csv",
    "commands.csv",
    "servo_telemetry.csv",
    "live_task_gate.json",
    "ipwm_closed_loop_evidence.json",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("schedule contains no rows")
    ids = [row["trial_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("trial_id values must be unique")
    return rows


def push_packet_errors(folder: Path) -> list[str]:
    errors = [name for name in sorted(REQUIRED_PUSH) if not (folder / name).is_file()]
    if not (folder / "ipwm_replan_cycles").is_dir():
        errors.append("ipwm_replan_cycles")
    if errors:
        return [f"missing:{name}" for name in errors]
    evidence = json.loads((folder / "ipwm_closed_loop_evidence.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS":
        errors.append("closed_loop_audit_not_PASS")
    if evidence.get("is_genuine_physical_ipwm_closed_loop_evidence") is not True:
        errors.append("not_genuine_physical_ipwm_closed_loop")
    return errors


def audit(protocol_path: Path, schedule_path: Path, evidence_root: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    rows = load_rows(schedule_path)
    details = []
    counts = Counter()
    by_task_condition: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        status = row["status"].strip().upper()
        counts[status] += 1
        key = f"{row['task']}:{row['condition']}"
        by_task_condition[key][status] += 1
        errors: list[str] = []
        folder = evidence_root / row["trial_id"]
        if status not in TERMINAL:
            errors.append("not_adjudicated")
        elif status in {"INVALID", "PREFLIGHT_EXCLUDED"}:
            adjudication_path = folder / "adjudication.json"
            if not adjudication_path.is_file():
                errors.append("missing:adjudication.json")
            else:
                decision = json.loads(adjudication_path.read_text(encoding="utf-8"))
                if decision.get("status") != status or not decision.get("reason"):
                    errors.append("invalid_adjudication")
                if not decision.get("evidence"):
                    errors.append("missing_adjudication_evidence")
                for item in decision.get("evidence", []):
                    import hashlib
                    source = folder / item["path"]
                    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item.get("sha256"):
                        errors.append("adjudication_evidence_missing_or_changed")
            if status == "PREFLIGHT_EXCLUDED" and row["priority"] == "core":
                errors.append("core_condition_cannot_be_preflight_excluded")
        elif status in {"GO", "NO_GO"}:
            if row["task"] == "push":
                errors.extend(push_packet_errors(folder))
            else:
                if not folder.is_dir():
                    errors.append("missing_grasp_trial_folder")
                for name in ("run_manifest.json", "daheng_FDE23080341_raw.avi",
                             "directshow_index1_raw.avi", "servo_telemetry.csv"):
                    if not (folder / name).is_file():
                        errors.append(f"missing:{name}")
        details.append({
            "trial_id": row["trial_id"], "task": row["task"],
            "condition": row["condition"], "status": status,
            "evidence_dir": str(folder.resolve()), "errors": errors,
        })

    core_push = [r for r in rows if r["task"] == "push" and r["priority"] == "core"]
    core_grasp = [r for r in rows if r["task"] == "grasp" and r["priority"] == "core"]
    required = core_push + core_grasp
    required_ids = {r["trial_id"] for r in required}
    required_details = [d for d in details if d["trial_id"] in required_ids]
    required_complete = all(d["status"] in {"GO", "NO_GO"} and not d["errors"]
                            for d in required_details)
    all_adjudicated = all(d["status"] in TERMINAL for d in details)
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "schedule_rows": len(rows),
        "status_counts": dict(counts),
        "by_task_condition": {k: dict(v) for k, v in sorted(by_task_condition.items())},
        "core_required_rows": len(required),
        "core_required_complete": required_complete,
        "all_rows_adjudicated": all_adjudicated,
        "evidence_ready_for_archive": required_complete and all_adjudicated
                                      and all(not d["errors"] for d in details),
        # This audit does not observe the final physical shutdown or verify
        # both backup trees. Packet completeness alone never authorizes teardown.
        "ready_to_dismantle": False,
        "closure_requirements_not_verified_here": [
            "two_backups_rehashed_against_source",
            "physical_safe_home_and_torque_off",
            "submission_tables_statistics_frames_video_index_failure_boundaries",
        ],
        "details": details,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=Path("config/experiment/real_ipwm_final_day_20260906.yaml"))
    parser.add_argument("--schedule", type=Path,
                        default=Path("data/real_robot/final_day_schedule_20260906.csv"))
    parser.add_argument("--evidence-root", type=Path,
                        default=Path("data/real_robot/session_20260901/final_day_trials"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/real_robot/final_day_completion_audit.json"))
    args = parser.parse_args()
    payload = audit(args.protocol, args.schedule, args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
