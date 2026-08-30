"""Stream object planar pose relative to a fixed table ArUco marker."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from robotarm.deployment.vision_pose import (
    object_in_reference, planar_pose, validate_camera_calibration, VelocityFilter)


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def required(config, key):
    value = config.get(key)
    if value is None:
        raise SystemExit(f"{key} is required; use measured calibration, not a placeholder")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(
        "config/deployment/eye_in_hand_aruco_v1.yaml"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        matrix, distortion, marker_size = validate_camera_calibration(
            required(config, "camera_matrix"),
            required(config, "distortion_coefficients"),
            required(config, "marker_size_m"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV ArUco is required: install opencv-contrib-python") from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit("installed OpenCV lacks ArUco; install opencv-contrib-python")
    dictionary_name = config["aruco_dictionary"]
    if not hasattr(cv2.aruco, dictionary_name):
        raise SystemExit(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    capture = cv2.VideoCapture(int(config["camera_index"]))
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera index {config['camera_index']}")
    output = Path(config["output_file"])
    velocity = VelocityFilter(float(config["velocity_smoothing"]),
                              float(config["maximum_detection_gap_s"]))
    reference_id, object_id = int(config["reference_marker_id"]), int(config["object_marker_id"])
    object_points = np.asarray([[-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0], [-.5, -.5, 0]],
                               np.float32)*marker_size
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            corners, ids, _ = detector.detectMarkers(frame)
            poses = {}
            if ids is not None:
                for corner, marker_id in zip(corners, ids.reshape(-1)):
                    solved, rvec, tvec = cv2.solvePnP(
                        object_points, corner.reshape(4, 2), matrix, distortion,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if solved:
                        poses[int(marker_id)] = (cv2.Rodrigues(rvec)[0], tvec.reshape(3))
            if reference_id in poses and object_id in poses:
                relative = object_in_reference(poses[reference_id][0], poses[reference_id][1],
                                               poses[object_id][1])
                xy = planar_pose(relative, tuple(config["planar_axes"]),
                                 tuple(config["planar_signs"]))
                now = time.time(); vel = velocity.update(xy, now)
                atomic_json(output, {"timestamp_unix_s": now,
                    "object_x_m": float(xy[0]), "object_y_m": float(xy[1]),
                    "object_vx_m_s": float(vel[0]), "object_vy_m_s": float(vel[1]),
                    "reference_marker_id": reference_id, "object_marker_id": object_id,
                    "source": "eye_in_hand_relative_aruco", "config_version": config["version"]})
            if args.show:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.imshow("BT-DPWM eye-in-hand pose", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
