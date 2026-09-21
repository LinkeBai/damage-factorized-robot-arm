"""Estimate a real-arm table-plane affine map from synchronized evidence.

The calibration pairs the overhead-camera gripper push-face pixel with the
forward-kinematic TCP XY computed from measured STS3215 joint feedback.  It is
strictly an evidence-derived bridge: rank-deficient motion, weak spatial
coverage, or excessive leave-one-trial-out error fails closed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from robotarm.analysis.yellow_cube_tracker import TrackerConfig, detect_yellow_cube
from robotarm.envs.fk import forward_kinematics


TRACKER = TrackerConfig(
    hsv_lower=(10, 80, 90), hsv_upper=(35, 255, 255), min_area_px=300,
    min_component_confidence=0.35, roi_xywh=(1000, 450, 350, 250),
)


def detect_distal_gripper_push_face(frame: np.ndarray) -> dict[str, object]:
    """Detect the rightmost red distal component without requiring the cube.

    The fixed overhead setup points the arm toward increasing image x.  We keep
    only red components in the frozen task ROI and select the component whose
    right edge is furthest downstream.  This is suitable for the small J1
    calibration sweep and avoids using the task object as a detector anchor.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array((0, 135, 75)), np.array((12, 255, 255)))
    red |= cv2.inRange(hsv, np.array((168, 135, 75)), np.array((179, 255, 255)))
    roi = np.zeros_like(red)
    roi[430:720, 760:1250] = 255
    red &= roi
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    components = [
        i for i in range(1, count)
        if 120 <= int(stats[i, cv2.CC_STAT_AREA]) <= 25000
    ]
    if not components:
        return {"status": "NOT_DETECTED", "component_count": 0}
    component = max(
        components,
        key=lambda i: int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
    )
    ys, xs = np.where(labels == component)
    points = np.column_stack((xs, ys)).astype(np.float64)
    max_x = float(np.max(points[:, 0]))
    distal = points[points[:, 0] >= max_x - 8.0]
    low_y, high_y = np.percentile(distal[:, 1], (10, 90))
    return {
        "status": "DETECTED",
        "component_count": len(components),
        "component_area_px": int(stats[component, cv2.CC_STAT_AREA]),
        "push_face_px": [max_x, float((low_y + high_y) / 2.0)],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def trial_samples(trial: Path, stride: int) -> list[dict[str, object]]:
    manifest = json.loads((trial / "run_manifest.json").read_text(encoding="utf-8"))
    videos = sorted(trial.glob("daheng_*_raw.avi"))
    if len(videos) != 1:
        raise ValueError(f"{trial}: expected one Daheng raw video, found {len(videos)}")
    frame_rows = [row for row in read_csv(trial / "frame_timestamps.csv")
                  if str(row.get("camera", "")).startswith("daheng")]
    frame_rows.sort(key=lambda row: int(row["frame_index"]))
    telemetry = read_csv(trial / "servo_telemetry.csv")
    if not frame_rows or not telemetry:
        raise ValueError(f"{trial}: missing timestamp or telemetry rows")
    telemetry_ns = np.asarray([
        (int(row["read_start_monotonic_ns"]) + int(row["read_end_monotonic_ns"])) / 2
        for row in telemetry
    ], dtype=np.float64)
    capture = cv2.VideoCapture(str(videos[0]))
    if not capture.isOpened():
        raise ValueError(f"{trial}: cannot decode {videos[0].name}")
    samples: list[dict[str, object]] = []
    try:
        for row in frame_rows:
            frame_index = int(row["frame_index"])
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride:
                continue
            cube = detect_yellow_cube(frame, TRACKER)
            gripper = detect_distal_gripper_push_face(frame)
            if gripper.get("status") != "DETECTED":
                continue
            timestamp = float(row["capture_mid_monotonic_ns"])
            telemetry_index = int(np.argmin(np.abs(telemetry_ns - timestamp)))
            feedback = telemetry[telemetry_index]
            q = np.asarray([float(feedback[f"j{i}_position_rad"]) for i in range(1, 6)])
            tcp = forward_kinematics(q)
            samples.append({
                "trial_id": manifest.get("trial_id", trial.name),
                "frame_index": frame_index,
                "capture_mid_monotonic_ns": int(timestamp),
                "telemetry_sample_index": int(feedback["sample_index"]),
                "sync_delta_ms": float(abs(telemetry_ns[telemetry_index] - timestamp) / 1e6),
                "tcp_base_xy_m": [float(tcp[0]), float(tcp[1])],
                "push_face_px": [float(x) for x in gripper["push_face_px"]],
                "cube_detected": bool(cube.detected),
            })
    finally:
        capture.release()
    return samples


def fit_affine(samples: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, float]]:
    base = np.asarray([row["tcp_base_xy_m"] for row in samples], dtype=np.float64)
    pixels = np.asarray([row["push_face_px"] for row in samples], dtype=np.float64)
    design = np.column_stack((base, np.ones(len(base))))
    matrix, _, rank, singular = np.linalg.lstsq(design, pixels, rcond=None)
    predicted = design @ matrix
    residual = np.linalg.norm(predicted - pixels, axis=1)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    return matrix, {
        "rank": int(rank),
        "condition_number": condition,
        "fit_rmse_px": float(np.sqrt(np.mean(residual ** 2))),
        "fit_p95_px": float(np.percentile(residual, 95)),
        "base_x_span_m": float(np.ptp(base[:, 0])),
        "base_y_span_m": float(np.ptp(base[:, 1])),
        "pixel_x_span": float(np.ptp(pixels[:, 0])),
        "pixel_y_span": float(np.ptp(pixels[:, 1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--maximum-rmse-px", type=float, default=5.0)
    parser.add_argument("--maximum-condition-number", type=float, default=1e5)
    parser.add_argument("--minimum-base-span-m", type=float, default=0.01)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be positive")
    samples: list[dict[str, object]] = []
    errors: list[str] = []
    for trial in args.trials:
        try:
            samples.extend(trial_samples(trial.resolve(), args.stride))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append(f"{trial}: {type(error).__name__}: {error}")
    diagnostics: dict[str, object] = {"sample_count": len(samples)}
    matrix = None
    if len(samples) >= 6:
        matrix, diagnostics = fit_affine(samples)
    unique_trials = sorted({str(row["trial_id"]) for row in samples})
    reasons = list(errors)
    if matrix is None:
        reasons.append("fewer_than_six_detected_correspondences")
    else:
        if diagnostics["rank"] != 3:
            reasons.append("affine_design_rank_is_not_three")
        if diagnostics["condition_number"] > args.maximum_condition_number:
            reasons.append("affine_design_is_ill_conditioned")
        if diagnostics["fit_rmse_px"] > args.maximum_rmse_px:
            reasons.append("fit_rmse_exceeds_gate")
        if min(diagnostics["base_x_span_m"], diagnostics["base_y_span_m"]) < args.minimum_base_span_m:
            reasons.append("two_dimensional_base_coverage_is_insufficient")
    passed = not reasons
    inverse = None
    if passed and matrix is not None:
        # [x,y,1] @ matrix = [u,v].  Invert the 2x2 linear block.
        linear = matrix[:2, :].T
        offset = matrix[2, :]
        inverse_linear = np.linalg.inv(linear)
        inverse = {
            "pixel_to_base_linear_m_per_px": inverse_linear.tolist(),
            "pixel_to_base_offset_m": (-offset @ inverse_linear.T).tolist(),
        }
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "scope": "overhead_pixel_to_nominal_fk_base_xy_affine",
        "trials": unique_trials,
        "samples": samples,
        "base_xy1_to_pixel_uv_matrix": None if matrix is None else matrix.tolist(),
        "inverse": inverse,
        "diagnostics": diagnostics,
        "gates": {
            "maximum_rmse_px": args.maximum_rmse_px,
            "maximum_condition_number": args.maximum_condition_number,
            "minimum_span_each_base_axis_m": args.minimum_base_span_m,
        },
        "failure_reasons": reasons,
        "claim_boundary": (
            "PASS supports a local planar task-frame bridge only; it does not "
            "identify actuator dynamics or authorize hardware motion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "trials", "diagnostics", "failure_reasons")}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
