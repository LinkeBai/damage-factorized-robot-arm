"""Finalize the versioned IPWM Push-only evidence inventory without editing raw trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from audit_real_ipwm_closed_loop_trial import audit_trial


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--shutdown", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(args.schedule.open(encoding="utf-8-sig", newline="")))
    ids = [row["trial_id"] for row in rows]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate_formal_trial_id")
    pending = [r["trial_id"] for r in rows if r["status"] not in {"GO", "NO_GO"}]
    if pending:
        errors.append("formal_schedule_has_pending_or_invalid_rows")
    audits, formal_rows = {}, []
    for row in rows:
        trial = args.trials / row["trial_id"]
        result = audit_trial(trial)
        audits[row["trial_id"]] = result
        if result.get("status") != "PASS" or result.get("task_outcome") != row["status"]:
            errors.append(f"formal_audit_mismatch:{row['trial_id']}")
        formal_rows.append({
            **row,
            "audit_status": result.get("status"),
            "task_outcome": result.get("task_outcome"),
            "replan_cycles": result.get("replan_cycles"),
            "endpoint_error_x_px": (result.get("endpoint_error_xy_px") or [None, None])[0],
            "endpoint_error_y_px": (result.get("endpoint_error_xy_px") or [None, None])[1],
            "maximum_consecutive_goal_frames": result.get("recomputed_maximum_consecutive_goal_frames"),
        })
    core = [r for r in formal_rows if r["priority"] == "core"]
    by_core = Counter(r["condition"] for r in core)
    if len(core) != 18 or by_core != Counter({k: 3 for k in ("intact", "D1", "D2", "D3", "D4", "D5")}):
        errors.append("core_membership_not_6x3")
    if any(r["task_outcome"] not in {"GO", "NO_GO"} for r in core):
        errors.append("core_contains_nonterminal_or_invalid")
    doubles = [r for r in formal_rows if r["priority"] == "double_fault"]
    if set(r["condition"] for r in doubles) != {"J1+J2", "J2+J3", "J3+J4"}:
        errors.append("double_fault_condition_missing")
    inventory_rows = list(csv.DictReader(args.inventory.open(encoding="utf-8-sig", newline="")))
    disk_ids = {p.name for p in args.trials.iterdir() if p.is_dir() and (p / "run_manifest.json").is_file()}
    inventory_ids = {r["trial_id"] for r in inventory_rows}
    if disk_ids != inventory_ids:
        errors.append("all_attempt_inventory_does_not_match_retained_trial_directories")
    shutdown = json.loads(args.shutdown.read_text(encoding="utf-8"))
    if not (shutdown.get("status") == "PASS" and shutdown.get("torque_off_verified") is True
            and shutdown.get("powered_hold_active") is False):
        errors.append("physical_shutdown_not_verified")

    csv_path = args.output / "formal_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(formal_rows[0]))
        writer.writeheader(); writer.writerows(formal_rows)

    frame_dir = args.output / "representative_frames"
    frame_dir.mkdir()
    representative = []
    chosen = {}
    for row in formal_rows:
        chosen.setdefault(row["condition"], row)
    for condition, row in chosen.items():
        video = args.trials / row["trial_id"] / "daheng_FDE23080341_raw.avi"
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        ok, frame = cap.read(); cap.release()
        if not ok:
            errors.append(f"representative_frame_decode_failed:{row['trial_id']}")
            continue
        target = frame_dir / f"{condition.replace('+', '_')}_{row['trial_id']}_final.jpg"
        if not cv2.imwrite(str(target), frame):
            errors.append(f"representative_frame_write_failed:{row['trial_id']}")
            continue
        representative.append({"condition": condition, "trial_id": row["trial_id"],
                               "selection": "last_decoded_overhead_frame",
                               "path": str(target.resolve()), "sha256": digest(target)})

    core_go = sum(r["task_outcome"] == "GO" for r in core)
    double_counts = defaultdict(Counter)
    for r in doubles:
        double_counts[r["condition"]][r["task_outcome"]] += 1
    md = ["# Final IPWM real-robot Push-only evidence", "",
          f"Core: **{core_go}/{len(core)} GO**, with three audit-valid trials for each of intact and D1-D5.", "",
          "| Double fault | GO | NO_GO | Valid n |", "|---|---:|---:|---:|"]
    for condition in ("J1+J2", "J2+J3", "J3+J4"):
        c = double_counts[condition]
        md.append(f"| {condition} | {c['GO']} | {c['NO_GO']} | {sum(c.values())} |")
    md += ["", "J3+J4 retained boundary: two audit-valid NO_GO runs followed by one audit-valid GO; all are reported.",
           "All invalid, aborted, and retry directories remain in the non-selective all-attempt inventory.",
           "Representative images are deterministic final decoded overhead frames, not hand-selected frames."]
    (args.output / "submission_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    payload = {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "formal_schedule": str(args.schedule.resolve()),
        "formal_schedule_sha256": digest(args.schedule),
        "formal_rows": len(rows), "core_rows": len(core), "core_go": core_go,
        "core_by_condition": dict(by_core),
        "double_fault_counts": {k: dict(v) for k, v in double_counts.items()},
        "pending_count": len(pending), "retained_trial_directory_count": len(disk_ids),
        "all_attempt_inventory_count": len(inventory_ids),
        "representative_frames": representative,
        "shutdown": shutdown,
        "errors": errors,
    }
    (args.output / "completion_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
