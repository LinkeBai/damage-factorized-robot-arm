"""Read-only readiness audit for BT-DPWM real-arm collection."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from robotarm.deployment.vision_pose import validate_camera_calibration


def result(ok, detail):
    return {"ok": bool(ok), "detail": str(detail)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-config", type=Path, default=Path(
        "config/deployment/eye_in_hand_aruco_calibrated.yaml"))
    parser.add_argument("--vision-pose-file", type=Path, default=Path(
        "runs/real_bt_dpwm_z70/live_vision_pose.json"))
    parser.add_argument("--maximum-vision-age-ms", type=float, default=150)
    parser.add_argument("--port", help="expected servo port; omit to accept any detected port")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = {}
    required_files = [Path("hardware/joint_map.yaml"), Path("hardware/safety_limits.yaml"),
                      Path("config/deployment/bt_dpwm_real_calibration_v1.yaml")]
    missing = [str(path) for path in required_files if not path.is_file()]
    checks["hardware_configuration"] = result(not missing, "present" if not missing else missing)
    try:
        from serial.tools import list_ports
        ports = [port.device for port in list_ports.comports()]
        port_ok = args.port in ports if args.port else bool(ports)
        checks["servo_serial"] = result(port_ok, {"detected": ports, "expected": args.port})
    except ImportError as exc:
        checks["servo_serial"] = result(False, f"pyserial unavailable: {exc}")
    try:
        import cv2
        checks["opencv_aruco"] = result(hasattr(cv2, "aruco"), cv2.__version__)
    except ImportError:
        checks["opencv_aruco"] = result(False, "install opencv-contrib-python")
    if args.vision_config.is_file():
        try:
            config = yaml.safe_load(args.vision_config.read_text(encoding="utf-8"))
            validate_camera_calibration(config.get("camera_matrix"),
                config.get("distortion_coefficients"), config.get("marker_size_m"))
            rms = float(config.get("calibration_rms_px", float("inf")))
            checks["camera_calibration"] = result(rms <= 1.0, f"RMS={rms:.3f}px")
        except (ValueError, TypeError, KeyError) as exc:
            checks["camera_calibration"] = result(False, exc)
    else:
        checks["camera_calibration"] = result(False, f"missing {args.vision_config}")
    if args.vision_pose_file.is_file():
        try:
            pose = json.loads(args.vision_pose_file.read_text(encoding="utf-8"))
            required = ("timestamp_unix_s", "object_x_m", "object_y_m",
                        "object_vx_m_s", "object_vy_m_s")
            age_ms = 1000*(time.time()-float(pose["timestamp_unix_s"]))
            finite = all(abs(float(pose[key])) < float("inf") for key in required)
            checks["live_vision_pose"] = result(
                finite and -50 <= age_ms <= args.maximum_vision_age_ms,
                f"age={age_ms:.1f}ms")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            checks["live_vision_pose"] = result(False, exc)
    else:
        checks["live_vision_pose"] = result(False, f"missing {args.vision_pose_file}")
    payload = {"ready_for_low_amplitude_smoke": all(x["ok"] for x in checks.values()),
               "motion_executed": False, "checks": checks}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if payload["ready_for_low_amplitude_smoke"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
