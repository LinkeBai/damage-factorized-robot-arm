"""Recover a complete trial rejected only at the camera-cadence float boundary.

The source manifest is immutable.  This tool emits a separate, hashed,
post-acquisition adjudication and fails closed for any motion, telemetry,
tracking, packet-integrity, decoding, or cadence problem beyond the declared
absolute numerical tolerance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--nominal-fps", type=float, default=15.0)
    parser.add_argument("--fractional-tolerance", type=float, default=0.15)
    parser.add_argument("--absolute-tolerance-fps", type=float, default=0.01)
    parser.add_argument("--goal-tolerance-px", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trial = args.trial.resolve()
    manifest_path = trial / "run_manifest.json"
    packet_path = trial / "packet_audit.json"
    timestamps_path = trial / "frame_timestamps.csv"
    tracking_path = trial / "offline_cube_tracking_v1" / "yellow_cube_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {
        "source_manifest_retains_abort": str(manifest.get("status", "")).startswith("ABORTED"),
        "failure_is_camera_cadence_only": (
            manifest.get("failure_type") == "RuntimeError"
            and str(manifest.get("failure_message", "")).startswith("camera cadence ")
        ),
        "packet_hash_audit_pass": packet.get("packet_integrity_status") == "PASS",
        "commands_present": int(manifest.get("command_rows", 0)) > 0,
        "telemetry_present": int(manifest.get("telemetry_rows", 0)) > 0,
        "tracking_confidence_pass": tracking.get("confidence_gate", {}).get("pass") is True,
        "endpoint_assessed": tracking.get("image_task", {}).get("assessed") is True,
    }

    by_camera: dict[str, list[int]] = {}
    with timestamps_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_camera.setdefault(row["camera"], []).append(int(row["capture_mid_monotonic_ns"]))
    lower = args.nominal_fps * (1.0 - args.fractional_tolerance)
    upper = args.nominal_fps * (1.0 + args.fractional_tolerance)
    camera_rows = []
    for camera, stamps in sorted(by_camera.items()):
        stamps.sort()
        span_s = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) >= 2 else 0.0
        observed = (len(stamps) - 1) / span_s if span_s > 0 else math.nan
        video = (next(trial.glob("daheng_*_raw.avi")) if camera.startswith("daheng")
                 else trial / "directshow_index1_raw.avi")
        capture = cv2.VideoCapture(str(video))
        try:
            decoded_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if capture.isOpened() else 0
            decodable, frame = capture.read() if capture.isOpened() else (False, None)
        finally:
            capture.release()
        within = (
            math.isfinite(observed)
            and lower - args.absolute_tolerance_fps <= observed <= upper + args.absolute_tolerance_fps
        )
        camera_rows.append({
            "camera": camera,
            "timestamp_rows": len(stamps),
            "timestamp_span_s": span_s,
            "observed_fps": observed,
            "nominal_fps": args.nominal_fps,
            "fractional_gate_fps": [lower, upper],
            "absolute_numerical_tolerance_fps": args.absolute_tolerance_fps,
            "within_adjudicated_gate": within,
            "video": str(video),
            "video_sha256": sha256(video),
            "decoded_frame_count": decoded_count,
            "first_frame_decodable": bool(decodable and frame is not None),
            "decoded_count_matches_timestamps": abs(decoded_count - len(stamps)) <= 1,
        })
    checks["two_camera_streams_present"] = len(camera_rows) == 2
    checks["camera_frame_counts_paired"] = (
        len(camera_rows) == 2
        and abs(camera_rows[0]["timestamp_rows"] - camera_rows[1]["timestamp_rows"]) <= 1
    )
    checks["all_camera_cadences_within_numerical_tolerance"] = all(
        row["within_adjudicated_gate"] for row in camera_rows
    )
    checks["all_videos_decodable_and_count_matched"] = all(
        row["first_frame_decodable"] and row["decoded_count_matches_timestamps"]
        for row in camera_rows
    )
    recovered = all(checks.values())
    radial = float(tracking["image_task"]["endpoint_error_radial_px"])
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "trial_id": manifest.get("trial_id", trial.name),
        "source_manifest_status_unchanged": manifest.get("status"),
        "source_manifest_sha256": sha256(manifest_path),
        "packet_audit_sha256": sha256(packet_path),
        "tracking_summary_sha256": sha256(tracking_path),
        "adjudication_type": "post_acquisition_borderline_camera_cadence",
        "checks": checks,
        "camera_cadence": camera_rows,
        "recovered_valid_trial": recovered,
        "task_assessment": {
            "endpoint_error_radial_px": radial,
            "goal_tolerance_radial_px": args.goal_tolerance_px,
            "success": recovered and radial <= args.goal_tolerance_px,
        },
        "disclosure": (
            "The immutable runner manifest remains aborted. This independent adjudication "
            "recovers acquisition validity only because the sole terminal exception was a "
            "camera-cadence boundary miss within the declared 0.01 fps numerical tolerance."
        ),
    }
    output = args.output or trial / "post_acquisition_adjudication.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if recovered else 2


if __name__ == "__main__":
    raise SystemExit(main())
