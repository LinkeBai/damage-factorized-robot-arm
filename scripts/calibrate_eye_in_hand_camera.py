"""Calibrate camera intrinsics from checkerboard images; never moves the robot."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml


def atomic_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True,
                        help="glob for checkerboard frames, e.g. calibration/camera/*.png")
    parser.add_argument("--columns", type=int, required=True,
                        help="checkerboard inner-corner columns")
    parser.add_argument("--rows", type=int, required=True,
                        help="checkerboard inner-corner rows")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--marker-size-m", type=float, required=True)
    parser.add_argument("--template", type=Path, default=Path(
        "config/deployment/eye_in_hand_aruco_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path(
        "config/deployment/eye_in_hand_aruco_calibrated.yaml"))
    parser.add_argument("--report", type=Path, default=Path(
        "runs/real_bt_dpwm_z70/camera_calibration_report.json"))
    args = parser.parse_args()
    if args.columns < 3 or args.rows < 3 or args.square_size_m <= 0 or args.marker_size_m <= 0:
        raise SystemExit("invalid checkerboard dimensions or physical sizes")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("install opencv-contrib-python in the project environment") from exc
    paths = sorted(glob.glob(args.images))
    if len(paths) < 10:
        raise SystemExit(f"need at least 10 diverse images; found {len(paths)}")
    pattern = (args.columns, args.rows)
    object_template = np.zeros((args.columns*args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_size_m
    object_points, image_points, used, image_size = [], [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is not None and image_size != gray.shape[::-1]:
            raise SystemExit("all calibration images must have the same resolution")
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(object_template.copy()); image_points.append(corners); used.append(path)
    if len(used) < 10:
        raise SystemExit(f"checkerboard detected in only {len(used)} images; need >=10")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    per_view = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        per_view.append(float(np.sqrt(np.mean((observed.reshape(-1, 2)-projected.reshape(-1, 2))**2))))
    if not np.isfinite(rms) or rms > 1.0:
        raise SystemExit(f"calibration RMS {rms:.3f}px exceeds frozen 1.0px gate")
    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    config.update(camera_matrix=matrix.tolist(),
                  distortion_coefficients=distortion.reshape(-1).tolist(),
                  marker_size_m=args.marker_size_m,
                  calibration_rms_px=float(rms))
    atomic_yaml(args.output, config)
    report = {"rms_px": float(rms), "mean_view_rmse_px": float(np.mean(per_view)),
              "maximum_view_rmse_px": float(np.max(per_view)), "images_used": used,
              "image_size": list(image_size), "checkerboard_inner_corners": list(pattern),
              "square_size_m": args.square_size_m, "output_config": str(args.output)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
