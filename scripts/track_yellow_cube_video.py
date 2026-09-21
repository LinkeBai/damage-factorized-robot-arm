"""Track the yellow cube in an existing raw video without touching hardware."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robotarm.analysis.yellow_cube_tracker import (  # noqa: E402
    CubeDetection,
    TrackerConfig,
    detect_yellow_cube,
    detection_to_row,
    draw_detection,
    load_frozen_calibration,
    sha256_file,
    summarize_rows,
)


CSV_FIELDS = (
    "frame_index", "video_time_s", "capture_mid_monotonic_ns", "host_utc",
    "detected", "confidence", "centroid_x_px", "centroid_y_px",
    "bbox_x_px", "bbox_y_px", "bbox_width_px", "bbox_height_px",
    "component_area_px", "bbox_extent", "mean_saturation", "mean_value",
    "centroid_x_m", "centroid_y_m",
)


def triplet(value: str) -> tuple[int, int, int]:
    parsed = tuple(int(part.strip()) for part in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("expected H,S,V")
    return parsed  # type: ignore[return-value]


def pair(value: str) -> tuple[float, float]:
    parsed = tuple(float(part.strip()) for part in value.split(","))
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("expected X,Y")
    return parsed  # type: ignore[return-value]


def roi(value: str) -> tuple[int, int, int, int]:
    parsed = tuple(int(part.strip()) for part in value.split(","))
    if len(parsed) != 4:
        raise argparse.ArgumentTypeError("expected X,Y,WIDTH,HEIGHT")
    return parsed  # type: ignore[return-value]


def resolve_video(source: Path) -> tuple[Path, Path | None]:
    source = source.resolve()
    if source.is_file():
        return source, None
    if not source.is_dir():
        raise FileNotFoundError(source)
    candidates = sorted(source.glob("*daheng*_raw.avi"))
    if len(candidates) != 1:
        raise ValueError(
            f"trial directory must contain exactly one Daheng raw AVI; found {candidates}")
    return candidates[0].resolve(), source


def timestamp_lookup(trial_dir: Path | None) -> tuple[dict[int, dict[str, str]], Path | None]:
    if trial_dir is None:
        return {}, None
    path = trial_dir / "frame_timestamps.csv"
    if not path.is_file():
        return {}, None
    lookup: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("camera", "").lower().startswith("daheng"):
                lookup[int(row["frame_index"])] = row
    return lookup, path


def manifest_metadata(trial_dir: Path | None) -> dict[str, Any] | None:
    if trial_dir is None:
        return None
    path = trial_dir / "run_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "trial_id": payload.get("trial_id"),
        "notice": payload.get("notice"),
        "evidence_scope": payload.get("evidence_scope"),
    }


def read_detection_at(video: Path, frame_index: int, config: TrackerConfig) -> tuple[Any, CubeDetection]:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame_index}")
        detection = detect_yellow_cube(frame, config)
        return frame, detection
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trial directory or existing video")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hsv-lower", type=triplet, default=(18, 90, 80))
    parser.add_argument("--hsv-upper", type=triplet, default=(42, 255, 255))
    parser.add_argument("--min-area-px", type=int, default=100)
    parser.add_argument("--max-area-fraction", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--roi", type=roi, help="Optional pixel ROI X,Y,WIDTH,HEIGHT")
    parser.add_argument("--endpoint-window-frames", type=int, default=15)
    parser.add_argument("--minimum-endpoint-detections", type=int, default=5)
    parser.add_argument("--minimum-detection-rate", type=float, default=0.90)
    parser.add_argument("--maximum-endpoint-mad-px", type=float, default=5.0)
    parser.add_argument("--stationary-threshold-px", type=float, default=2.0)
    parser.add_argument("--image-direction", type=pair, default=(1.0, 0.0),
                        help="Image-space measurement direction X,Y (+x right, +y down)")
    parser.add_argument("--task-goal-px", type=pair,
                        help="Frozen image-space task-goal center X,Y")
    parser.add_argument("--task-goal-box-wh-px", type=pair,
                        help="Frozen task-goal box width,height; requires --task-goal-px")
    parser.add_argument("--task-goal-tolerance-px", type=float,
                        help="Optional frozen radial endpoint-error success tolerance in pixels")
    parser.add_argument("--calibration", type=Path,
                        help="Optional JSON with frozen=true and scale/homography")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--write-endpoint-images", action="store_true")
    args = parser.parse_args()
    if (args.task_goal_px is None) != (args.task_goal_box_wh_px is None):
        parser.error("--task-goal-px and --task-goal-box-wh-px must be supplied together")
    if args.task_goal_box_wh_px is not None and min(args.task_goal_box_wh_px) <= 0:
        parser.error("task-goal box dimensions must be positive")
    if args.task_goal_tolerance_px is not None:
        if args.task_goal_px is None:
            parser.error("--task-goal-tolerance-px requires --task-goal-px")
        if not math.isfinite(args.task_goal_tolerance_px) or args.task_goal_tolerance_px <= 0:
            parser.error("task-goal tolerance must be finite and positive")

    video, trial_dir = resolve_video(args.input)
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  ((trial_dir / "offline_yellow_cube_tracking").resolve()
                   if trial_dir else
                   (video.parent / f"{video.stem}_yellow_cube_tracking").resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "yellow_cube_per_frame.csv"
    summary_path = output_dir / "yellow_cube_summary.json"
    overlay_path = output_dir / "yellow_cube_overlay.avi"
    if video == csv_path or video == summary_path or video == overlay_path:
        raise SystemExit("output path must not overwrite the source video")

    config = TrackerConfig(
        hsv_lower=args.hsv_lower,
        hsv_upper=args.hsv_upper,
        min_area_px=args.min_area_px,
        max_area_fraction=args.max_area_fraction,
        min_component_confidence=args.min_confidence,
        endpoint_window_frames=args.endpoint_window_frames,
        minimum_endpoint_detections=args.minimum_endpoint_detections,
        minimum_detection_rate=args.minimum_detection_rate,
        maximum_endpoint_mad_px=args.maximum_endpoint_mad_px,
        stationary_threshold_px=args.stationary_threshold_px,
        roi_xywh=args.roi,
    )
    config.validate()
    calibration = (load_frozen_calibration(args.calibration.resolve())
                   if args.calibration else None)
    timestamps, timestamps_path = timestamp_lookup(trial_dir)
    source_hash_before = sha256_file(video)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise SystemExit("video reports a non-positive FPS")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.write_overlay:
        writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
            (width, height))
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"could not create overlay video: {overlay_path}")
    rows: list[dict[str, Any]] = []
    previous_centroid = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detection = detect_yellow_cube(frame, config, previous_centroid)
            if detection.detected:
                previous_centroid = detection.centroid
            row = detection_to_row(frame_index, frame_index / fps,
                                   detection, calibration)
            timing = timestamps.get(frame_index, {})
            row["capture_mid_monotonic_ns"] = (
                int(timing["capture_mid_monotonic_ns"])
                if timing.get("capture_mid_monotonic_ns") else None)
            row["host_utc"] = timing.get("host_utc") or None
            rows.append(row)
            if writer is not None:
                writer.write(draw_detection(frame, detection))
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise SystemExit("video contained no readable frames")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    summary = summarize_rows(rows, config, args.image_direction, calibration)
    image_task = None
    if args.task_goal_px is not None:
        goal_x, goal_y = args.task_goal_px
        goal_w, goal_h = args.task_goal_box_wh_px
        end_median = summary["end_px"]["median"]
        assessed = bool(summary["confidence_gate"]["pass"] and end_median is not None)
        if assessed:
            error_x = float(end_median[0]) - goal_x
            error_y = float(end_median[1]) - goal_y
            error_radial = math.hypot(error_x, error_y)
            success = (
                error_radial <= args.task_goal_tolerance_px
                if args.task_goal_tolerance_px is not None
                else abs(error_x) <= goal_w / 2 and abs(error_y) <= goal_h / 2
            )
        else:
            error_x = error_y = error_radial = None
            success = None
        image_task = {
            "coordinate_system": "overhead_image_px_(+x_right,+y_down)",
            "goal_center_px": [goal_x, goal_y],
            "goal_box_wh_px": [goal_w, goal_h],
            "goal_tolerance_radial_px": args.task_goal_tolerance_px,
            "endpoint_error_xy_px": [error_x, error_y] if assessed else None,
            "endpoint_error_radial_px": error_radial,
            "assessed": assessed,
            "success": success,
            "criterion": (
                f"robust final centroid radial error <= {args.task_goal_tolerance_px:g} px"
                if args.task_goal_tolerance_px is not None
                else "robust final centroid lies inside the frozen axis-aligned goal box"
            ),
        }
        summary["image_task"] = image_task
    endpoint_artifacts: dict[str, str] = {}
    if args.write_endpoint_images:
        for label, endpoint in (("start", summary["start_px"]),
                                ("end", summary["end_px"])):
            frame_id = endpoint["representative_frame_index"]
            if frame_id is None:
                continue
            frame, detection = read_detection_at(video, frame_id, config)
            path = output_dir / f"yellow_cube_{label}_overlay.png"
            annotated = draw_detection(frame, detection)
            if args.task_goal_px is not None:
                gx, gy = args.task_goal_px
                gw, gh = args.task_goal_box_wh_px
                goal_color = ((0, 255, 0) if image_task and image_task["success"]
                              else (255, 0, 255))
                cv2.rectangle(annotated,
                              (round(gx-gw/2), round(gy-gh/2)),
                              (round(gx+gw/2), round(gy+gh/2)),
                              goal_color, 4)
                if args.task_goal_tolerance_px is not None:
                    cv2.circle(annotated, (round(gx), round(gy)),
                               round(args.task_goal_tolerance_px), goal_color, 3,
                               cv2.LINE_AA)
                cv2.putText(annotated, "TASK SUCCESS" if image_task and image_task["success"] else "TASK GOAL",
                            (round(gx+gw/2+8), round(gy)),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, goal_color, 2,
                            cv2.LINE_AA)
            if not cv2.imwrite(str(path), annotated):
                raise RuntimeError(f"could not write image: {path}")
            endpoint_artifacts[label] = str(path)

    source_hash_after = sha256_file(video)
    artifact_hashes = {str(csv_path): sha256_file(csv_path)}
    if args.write_overlay:
        artifact_hashes[str(overlay_path)] = sha256_file(overlay_path)
    for path_string in endpoint_artifacts.values():
        artifact_hashes[path_string] = sha256_file(Path(path_string))
    summary.update({
        "source": {
            "video": str(video),
            "size_bytes": video.stat().st_size,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "unchanged": source_hash_before == source_hash_after,
            "fps": fps,
            "width": width,
            "height": height,
            "timestamps_csv": str(timestamps_path.resolve()) if timestamps_path else None,
            "timestamps_sha256": sha256_file(timestamps_path) if timestamps_path else None,
            "run_manifest": manifest_metadata(trial_dir),
        },
        "tracker_config": asdict(config),
        "artifacts": {
            "per_frame_csv": str(csv_path),
            "overlay_video": str(overlay_path) if args.write_overlay else None,
            "endpoint_images": endpoint_artifacts,
            "sha256": artifact_hashes,
        },
        "audit_notice": (
            "Offline observation only. The raw source was opened read-only and was "
            "not rewritten. This tracker cannot convert an aborted acquisition into "
            "a valid task trial or infer physical task success."
        ),
    })
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "source_unchanged": summary["source"]["unchanged"],
        "detection_rate": summary["detection_rate"],
        "displacement_px": summary["displacement_px"],
        "projected_displacement_px": summary["projected_displacement_px"],
        "basically_unmoved_in_pixel_space": summary["basically_unmoved_in_pixel_space"],
        "metric_displacement_m": summary["metric"]["displacement_m"],
        "confidence_gate_pass": summary["confidence_gate"]["pass"],
        "source_run_status": (
            summary["source"]["run_manifest"] or {}).get("status"),
        "image_task": image_task,
    }, indent=2))


if __name__ == "__main__":
    main()
