"""Create a frozen Level-A session manifest from real on-site identifiers."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data/real_robot/session_manifest_template.yaml"
SAFETY = ROOT / "hardware/safety_limits.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is not an existing file: {path}")


def require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{name} is not an existing directory: {path}")


def build(args) -> dict:
    for path, name in (
        (args.schedule, "schedule"), (args.trajectory_library, "trajectory library"),
        (args.trajectory_validation, "trajectory validation"),
        (args.left_calibration, "left calibration"),
        (args.horizontal_calibration, "horizontal calibration"),
        (args.synchronization_video, "synchronization video"),
    ):
        require_file(path, name)
    for path, name in (
        (args.left_video_directory, "left video directory"),
        (args.horizontal_video_directory, "horizontal video directory"),
        (args.control_log_directory, "control log directory"),
        (args.backup_copy_1, "backup copy 1"),
        (args.backup_copy_2, "backup copy 2"),
    ):
        require_dir(path, name)
    validation = yaml.safe_load(args.trajectory_validation.read_text(encoding="utf-8")) \
        if args.trajectory_validation.suffix.lower() in {".yaml", ".yml"} else None
    if validation is None:
        import json
        validation = json.loads(args.trajectory_validation.read_text(encoding="utf-8"))
    library_hash = digest(args.trajectory_library)
    if validation.get("status") != "PASS" or validation.get("library_sha256") != library_hash:
        raise ValueError("trajectory validation must PASS and match the trajectory-library hash")
    safety = yaml.safe_load(SAFETY.read_text(encoding="utf-8"))
    maximum_speed_deg_s = min(float(item["max_speed_deg_s"]) for item in safety["joints"])
    maximum_lock_error_deg = float(safety["damage_test"]["max_lock_drift_deg"])
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload.update({
        "status": "frozen_before_level_a_trials",
        "session_id": args.session_id,
        "date_local": args.date_local,
        "operator": args.operator,
    })
    payload["hardware"].update({
        "robot_asset_id": args.robot_asset_id,
        "gripper_asset_id": args.gripper_asset_id,
        "block_asset_id": args.block_asset_id,
        "emergency_stop_checked": True,
        "joint_direction_check_complete": True,
        "low_speed_stop_check_complete": True,
    })
    payload["cameras"].update({
        "left_eye_to_hand_serial": args.left_camera_serial,
        "horizontal_eye_to_hand_serial": args.horizontal_camera_serial,
        "left_calibration_file": str(args.left_calibration.resolve()),
        "horizontal_calibration_file": str(args.horizontal_calibration.resolve()),
        "synchronization_event_video": str(args.synchronization_video.resolve()),
    })
    payload["safety"].update({
        "maximum_commanded_joint_speed_rad_s": float(np.radians(maximum_speed_deg_s)),
        "maximum_allowed_lock_error_rad": float(np.radians(maximum_lock_error_deg)),
        "workspace_boundary_description": args.workspace_boundary,
    })
    payload["randomization"].update({
        "schedule_file": str(args.schedule.resolve()),
        "schedule_sha256_before_trials": digest(args.schedule),
        "physical_reset_fixture_description": args.reset_fixture,
        "action_library_hash": library_hash,
        "action_library_file": str(args.trajectory_library.resolve()),
        "action_library_validation_file": str(args.trajectory_validation.resolve()),
        "action_interface_bridge_file": "",
        "action_interface_validation_file": "",
        "learned_method_comparison_authorized": False,
    })
    payload["data_roots"].update({
        "left_video_directory": str(args.left_video_directory.resolve()),
        "horizontal_video_directory": str(args.horizontal_video_directory.resolve()),
        "control_log_directory": str(args.control_log_directory.resolve()),
        "backup_copy_1": str(args.backup_copy_1.resolve()),
        "backup_copy_2": str(args.backup_copy_2.resolve()),
    })
    payload["freeze_record"].update({
        "frozen_before_first_method_trial": True,
        "freeze_timestamp_local": args.freeze_timestamp,
        "operator_signature_or_initials": args.operator_initials,
    })
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("session-id", "date-local", "operator", "operator-initials",
                 "robot-asset-id", "gripper-asset-id", "block-asset-id",
                 "left-camera-serial", "horizontal-camera-serial",
                 "workspace-boundary", "reset-fixture"):
        result.add_argument(f"--{name}", required=True)
    for name in ("schedule", "trajectory-library", "trajectory-validation",
                 "left-calibration", "horizontal-calibration",
                 "synchronization-video", "left-video-directory",
                 "horizontal-video-directory", "control-log-directory",
                 "backup-copy-1", "backup-copy-2", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--freeze-timestamp",
                        default=datetime.now().astimezone().isoformat(timespec="seconds"))
    return result


def main() -> None:
    args = parser().parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
