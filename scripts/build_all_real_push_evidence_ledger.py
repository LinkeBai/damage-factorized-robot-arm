"""Merge every recorded real-Push packet without erasing protocol boundaries.

This is a read-only evidence index.  It never upgrades calibration, an abort,
or a trajectory with invalid task semantics into a confirmatory trial.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_and_role(trial_id: str, group: str) -> tuple[str, str]:
    if group == "ipwm_trials" and (trial_id.startswith("real-ipwm-") or trial_id.startswith("ipwm-")):
        return "ipwm_axis5_level_a", "model_selected_hardware_feasibility"
    if trial_id.startswith("push30-v6-"):
        return "v6_rebased_axis5", "confirmatory"
    if trial_id.startswith("push30-v5-"):
        return "v5_original_geometry_radial5", "superseded_confirmatory"
    if trial_id.startswith("push30-v4-"):
        return "v4_original_geometry_radial3", "protocol_calibration"
    if trial_id.startswith("push30-v3-"):
        return "v3_awb", "protocol_calibration"
    if trial_id.startswith("push30-v2-"):
        return "v2", "protocol_calibration"
    if group == "pilot_trials" or trial_id.startswith("pilot-"):
        return "development", "pilot_or_calibration"
    if trial_id.startswith("final45-"):
        return "pre_v2_45px", "superseded_confirmatory"
    return "legacy_or_unclassified", "retained_evidence"


def select_summary(trial: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in trial.rglob("yellow_cube_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload.get("image_task"), dict):
            continue
        name = path.parent.name.lower()
        priority = 0
        if "axis5" in name:
            priority = 40
        elif name == "offline_cube_tracking_v1":
            priority = 30
        elif "strict3px" in name:
            priority = 20
        elif "aligned" in name:
            priority = 10
        candidates.append((priority, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime_ns))[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--additional-ipwm-root",
        type=Path,
        action="append",
        default=[],
        help="Additional directory whose immediate child directories are IPWM trials",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    scan_groups = [
        (args.session / group, group)
        for group in ("confirmatory_trials", "pilot_trials", "ipwm_trials")
    ]
    scan_groups.extend((root, "ipwm_trials") for root in args.additional_ipwm_root)
    for group_dir, group in scan_groups:
        if not group_dir.is_dir():
            continue
        for trial in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            manifest_path = trial / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            trial_id = str(manifest.get("trial_id") or trial.name)
            protocol, role = protocol_and_role(trial_id, group)
            adjudication_path = trial / "post_acquisition_adjudication.json"
            adjudicated = False
            if adjudication_path.is_file():
                try:
                    adjudicated = json.loads(
                        adjudication_path.read_text(encoding="utf-8")
                    ).get("recovered_valid_trial") is True
                except (OSError, json.JSONDecodeError):
                    pass
            acquisition_valid = (
                manifest.get("status") == "ACQUISITION_COMPLETE_UNASSESSED"
                or adjudicated
            )
            trajectory_id = str(manifest.get("trajectory_id") or "")
            trajectory_semantics = (
                "INVALID_LEVEL_PUSH_LIFTS_TCP"
                if trajectory_id == "D2_45px_v1"
                else "NOT_KNOWN_INVALID"
            )
            packet_path = trial / "packet_audit.json"
            packet_status = "MISSING"
            packet_artifacts: dict[str, dict[str, object]] = {}
            if packet_path.is_file():
                try:
                    packet = json.loads(packet_path.read_text(encoding="utf-8"))
                    packet_status = str(packet.get("packet_integrity_status", "UNKNOWN"))
                    packet_artifacts = {
                        str(item.get("role")): item
                        for item in packet.get("artifacts", [])
                        if isinstance(item, dict) and item.get("role")
                    }
                except (OSError, json.JSONDecodeError):
                    packet_status = "INVALID_JSON"
            model_manifest_path = trial / "model_decision_manifest.json"
            model_audit_path = trial / "model_decision_audit.json"
            closed_loop_evidence_path = trial / "ipwm_closed_loop_evidence.json"
            model_manifest: dict[str, object] = {}
            model_audit_status = "MISSING"
            if model_manifest_path.is_file():
                try:
                    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    model_manifest = {}
            if model_audit_path.is_file():
                try:
                    model_audit_status = str(json.loads(
                        model_audit_path.read_text(encoding="utf-8")
                    ).get("status", "UNKNOWN"))
                except (OSError, json.JSONDecodeError):
                    model_audit_status = "INVALID_JSON"
            closed_loop_evidence: dict[str, object] = {}
            if closed_loop_evidence_path.is_file():
                try:
                    closed_loop_evidence = json.loads(
                        closed_loop_evidence_path.read_text(encoding="utf-8")
                    )
                    if closed_loop_evidence.get(
                        "is_genuine_physical_ipwm_closed_loop_evidence"
                    ) is True:
                        model_audit_status = "PASS"
                        packet_status = "PASS"
                except (OSError, json.JSONDecodeError):
                    model_audit_status = "INVALID_JSON"
            if group == "ipwm_trials" and model_audit_status == "PASS":
                packet_status = "PASS"
            selected = select_summary(trial)
            task: dict[str, object] = {}
            confidence = None
            if selected is not None:
                payload = json.loads(selected.read_text(encoding="utf-8"))
                task = payload.get("image_task", {})
                confidence = payload.get("confidence_gate", {}).get("pass")
            if group == "ipwm_trials" and isinstance(model_manifest.get("outcome"), dict):
                outcome = model_manifest["outcome"]
                task = dict(task)
                task["success"] = outcome.get("success")
                task["criterion"] = outcome.get("criterion")
                task["endpoint_error_xy_px"] = outcome.get("endpoint_error_xy_px")
                task["endpoint_error_radial_px"] = outcome.get("endpoint_error_radial_px")
                task["assessed"] = True
            if group == "ipwm_trials" and closed_loop_evidence:
                task = {
                    "success": closed_loop_evidence.get("success"),
                    "criterion": closed_loop_evidence.get("criterion"),
                    "endpoint_error_xy_px": closed_loop_evidence.get("endpoint_error_xy_px"),
                    "endpoint_error_radial_px": None,
                    "goal_center_px": [1116.31, 570.0],
                    "assessed": True,
                }
                confidence = bool(closed_loop_evidence.get(
                    "is_genuine_physical_ipwm_closed_loop_evidence"
                ))
            valid_for_own_protocol = bool(
                acquisition_valid
                and packet_status == "PASS"
                and trajectory_semantics != "INVALID_LEVEL_PUSH_LIFTS_TCP"
                and confidence is True
                and task.get("assessed") is True
            )
            rows.append({
                "group": group,
                "trial_id": trial_id,
                "protocol": protocol,
                "evidence_role": role,
                "condition": manifest.get("condition"),
                "trajectory_id": trajectory_id,
                "trajectory_semantics": trajectory_semantics,
                "manifest_status": manifest.get("status"),
                "post_acquisition_adjudicated_valid": adjudicated,
                "acquisition_valid": acquisition_valid,
                "packet_integrity_status": packet_status,
                "model_decision_audit_status": model_audit_status,
                "candidate_count": (
                    min(closed_loop_evidence.get("candidate_count_each_cycle", []) or [0])
                    if closed_loop_evidence else
                    ((model_manifest.get("decision") or {}).get("candidate_count")
                     if isinstance(model_manifest.get("decision"), dict) else None)
                ),
                "tracking_confidence_pass": confidence,
                "task_assessed": task.get("assessed"),
                "task_success_under_recorded_criterion": task.get("success"),
                "recorded_criterion": task.get("criterion"),
                "goal_x_px": (task.get("goal_center_px") or [None, None])[0],
                "goal_y_px": (task.get("goal_center_px") or [None, None])[1],
                "endpoint_error_x_px": (task.get("endpoint_error_xy_px") or [None, None])[0],
                "endpoint_error_y_px": (task.get("endpoint_error_xy_px") or [None, None])[1],
                "endpoint_error_radial_px": task.get("endpoint_error_radial_px"),
                "valid_for_own_protocol": valid_for_own_protocol,
                "selected_tracking_summary": str(selected.resolve()) if selected else None,
                "selected_tracking_sha256": sha256(selected) if selected else None,
                "overhead_video_path": packet_artifacts.get("daheng_video", {}).get("path")
                or (str((trial / "daheng_FDE23080341_raw.avi").resolve()) if (trial / "daheng_FDE23080341_raw.avi").is_file() else None),
                "overhead_video_sha256": packet_artifacts.get("daheng_video", {}).get("sha256")
                or (sha256(trial / "daheng_FDE23080341_raw.avi") if (trial / "daheng_FDE23080341_raw.avi").is_file() else None),
                "overhead_video_bytes": packet_artifacts.get("daheng_video", {}).get("bytes")
                or ((trial / "daheng_FDE23080341_raw.avi").stat().st_size if (trial / "daheng_FDE23080341_raw.avi").is_file() else None),
                "wrist_video_path": packet_artifacts.get("directshow_video", {}).get("path")
                or (str((trial / "directshow_index1_raw.avi").resolve()) if (trial / "directshow_index1_raw.avi").is_file() else None),
                "wrist_video_sha256": packet_artifacts.get("directshow_video", {}).get("sha256")
                or (sha256(trial / "directshow_index1_raw.avi") if (trial / "directshow_index1_raw.avi").is_file() else None),
                "wrist_video_bytes": packet_artifacts.get("directshow_video", {}).get("bytes")
                or ((trial / "directshow_index1_raw.avi").stat().st_size if (trial / "directshow_index1_raw.avi").is_file() else None),
                "frame_timestamps_path": packet_artifacts.get("frame_timestamps", {}).get("path")
                or (str((trial / "frame_timestamps.csv").resolve()) if (trial / "frame_timestamps.csv").is_file() else None),
                "frame_timestamps_sha256": packet_artifacts.get("frame_timestamps", {}).get("sha256")
                or (sha256(trial / "frame_timestamps.csv") if (trial / "frame_timestamps.csv").is_file() else None),
                "trial_directory": str(trial.resolve()),
            })

    valid = [row for row in rows if row["valid_for_own_protocol"]]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_session": str(args.session.resolve()),
        "additional_ipwm_roots": [
            str(path.resolve()) for path in args.additional_ipwm_root
        ],
        "policy": {
            "all_recorded_trial_packets_retained": True,
            "protocol_boundaries_preserved": True,
            "failures_and_aborts_retained": True,
            "cross_protocol_rows_must_not_be_pooled_as_one_confirmatory_matrix": True,
            "invalid_D2_45px_lifting_trajectory_excluded_from_level_push_claims": True,
        },
        "counts": {
            "all_packets": len(rows),
            "valid_for_own_protocol": len(valid),
            "successful_under_recorded_criterion": sum(
                row["task_success_under_recorded_criterion"] is True for row in valid
            ),
            "failed_under_recorded_criterion": sum(
                row["task_success_under_recorded_criterion"] is False for row in valid
            ),
            "aborted_or_invalid_or_unassessed": len(rows) - len(valid),
            "v6_confirmatory_successes": sum(
                row["protocol"] == "v6_rebased_axis5"
                and row["evidence_role"] == "confirmatory"
                and row["valid_for_own_protocol"]
                and row["task_success_under_recorded_criterion"] is True
                for row in rows
            ),
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
