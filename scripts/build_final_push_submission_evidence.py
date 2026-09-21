"""Reaudit every retained attempt and produce a preliminary, non-selective index.

This never edits raw packets, schedule rows, or existing adjudications. Missing
and unassigned attempts stay visible. Formal membership needs separate review.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_real_robot_trial_packet import audit_trial_packet
from scripts.audit_real_ipwm_closed_loop_trial import audit_trial, sha256


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_endpoints(video, output, prefix):
    cap = cv2.VideoCapture(str(video))
    records = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for label, index in (("first", 0), ("last", count - 1)):
            if index < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            path = output / f"{prefix}_{label}.png"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"could not save {path}")
            records.append({"frame_index": index, "image": str(path.resolve()),
                            "sha256": sha256(path)})
    finally:
        cap.release()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", type=Path, default=ROOT / "data/real_robot/final_day_schedule_20260906.csv")
    ap.add_argument("--trials", type=Path, default=ROOT / "data/real_robot/session_20260901/final_day_trials")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audits = args.output / "audits"; audits.mkdir()
    frames = args.output / "frames"; frames.mkdir()
    scheduled = list(csv.DictReader(args.schedule.open(encoding="utf-8-sig", newline="")))
    by_id = {row["trial_id"]: row for row in scheduled}
    # Scan physical directories as well as the schedule: retries cannot vanish.
    ids = sorted(set(by_id) | {p.name for p in args.trials.iterdir() if p.is_dir()})
    rows, videos = [], []
    for trial_id in ids:
        folder = args.trials / trial_id
        declared = by_id.get(trial_id, {})
        row = {"trial_id": trial_id, "scheduled": trial_id in by_id,
               "declared_status": declared.get("status", "UNASSIGNED_ATTEMPT"),
               "condition": declared.get("condition", ""),
               "packet_status": "MISSING", "loop_status": "NOT_ASSESSED",
               "observed_task_outcome": "NOT_ASSESSED", "endpoint_error_px": "",
               "replan_cycles": "", "minimum_candidate_count": "",
               "failure_reason": "", "formal_membership": "NOT_VERIFIED"}
        if not (folder / "run_manifest.json").is_file():
            rows.append(row)
            continue
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8-sig"))
        row["condition"] = manifest.get("condition", row["condition"])
        row["failure_reason"] = manifest.get("failure_message", "")
        packet = audit_trial_packet(folder, expected_trial_id=trial_id,
                                    expected_condition=declared.get("condition"))
        loop = audit_trial(folder)
        for label, result in (("packet", packet), ("loop", loop)):
            (audits / f"{trial_id}_{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        row["packet_status"] = packet["status"]
        row["loop_status"] = loop["status"]
        outcome = loop.get("task_outcome", "INVALID")
        if not packet.get("valid_trial") or loop["status"] != "PASS":
            outcome = "INVALID"
        row["observed_task_outcome"] = outcome
        error = loop.get("endpoint_error_xy_px")
        if error and all(v is not None and math.isfinite(v) for v in error):
            row["endpoint_error_px"] = math.hypot(*error)
        row["replan_cycles"] = loop.get("replan_cycles", "")
        counts = loop.get("candidate_count_each_cycle", [])
        row["minimum_candidate_count"] = min(counts) if counts else ""
        if not row["failure_reason"] and outcome == "INVALID":
            row["failure_reason"] = "; ".join(packet.get("errors", []) + loop.get("errors", []))
        rows.append(row)
        for video in sorted(folder.glob("*_raw.avi")):
            videos.append({"trial_id": trial_id, "path": str(video.resolve()),
                           "bytes": video.stat().st_size, "sha256": sha256(video),
                           "endpoints": export_endpoints(video, frames, f"{trial_id}_{video.stem}")})
        print(f"{trial_id}: packet={packet['status']} loop={loop['status']} outcome={outcome}", flush=True)
    write_csv(args.output / "all_attempts.csv", rows, list(rows[0]))
    counts = Counter(row["observed_task_outcome"] for row in rows)
    summary = {"status": "PRELIMINARY_NOT_SUBMISSION_READY", "goal_complete": False,
               "schedule_sha256": sha256(args.schedule), "counts_including_missing": dict(counts),
               "formal_success_rate": None, "formal_valid_n": None,
               "reason": "Repeat/start-variant and frozen execution membership require independent verification; do not pool unassigned retries.",
               "videos": videos}
    (args.output / "video_index_and_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Preliminary IPWM Push evidence inventory", "",
             "NOT SUBMISSION READY. No formal success rate is reported. All retained attempts, including unscheduled retries, are indexed.", "",
             "| Attempt | Condition | Packet | Loop audit | Outcome | Endpoint error (px) |",
             "|---|---|---|---|---|---:|"]
    for row in rows:
        error = row["endpoint_error_px"]
        shown = f"{error:.3f}" if isinstance(error, float) else "—"
        lines.append(f"| {row['trial_id']} | {row['condition']} | {row['packet_status']} | {row['loop_status']} | {row['observed_task_outcome']} | {shown} |")
    lines += ["", "## Interpretation limits", "",
              "- Repeated attempts are not independent repetitions and are not silently substituted for a scheduled row.",
              "- Endpoint images are deterministic first/last decoded frames, not a selection of successful scenes.",
              "- Packet and loop checks do not independently verify frozen start-variant assignment or all calibration requirements.",
              "- Historical 30/45 px open-loop and the September 4 J1+J2 pipeline test are outside this table.",
              "- Missing runs, physical failure, invalid acquisition, and preflight exclusion are distinct; no missing metric is filled with zero."]
    (args.output / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
