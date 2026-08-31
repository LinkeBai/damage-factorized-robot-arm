import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.prepare_real_robot_level_a_session import build
from scripts.generate_real_robot_level_a_schedule import FIELDS, build as build_schedule


def test_session_builder_freezes_known_safety_and_hashes(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.csv"
    with schedule.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        writer.writerows(build_schedule(1, 10, {"intact": "i", "D2": "d2", "D3": "d3"}))
    library = tmp_path / "library.csv"; library.write_text("library", encoding="utf-8")
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"status": "PASS", "library_sha256": library_hash}), encoding="utf-8")
    files = [tmp_path / name for name in ("left.yaml", "horizontal.yaml", "sync.mp4")]
    for item in files: item.write_text("x", encoding="utf-8")
    dirs = [tmp_path / name for name in ("left", "horizontal", "logs", "b1", "b2")]
    for item in dirs: item.mkdir()
    args = argparse.Namespace(
        schedule=schedule, trajectory_library=library, trajectory_validation=validation,
        left_calibration=files[0], horizontal_calibration=files[1],
        synchronization_video=files[2], left_video_directory=dirs[0],
        horizontal_video_directory=dirs[1], control_log_directory=dirs[2],
        backup_copy_1=dirs[3], backup_copy_2=dirs[4], session_id="S1",
        date_local="2026-09-01", operator="Operator", operator_initials="OP",
        robot_asset_id="R", gripper_asset_id="G", block_asset_id="B",
        left_camera_serial="L", horizontal_camera_serial="H",
        workspace_boundary="marked rectangle", reset_fixture="three marks",
        freeze_timestamp="2026-09-01T08:00:00+08:00")
    payload = build(args)
    assert payload["randomization"]["schedule_sha256_before_trials"] == hashlib.sha256(schedule.read_bytes()).hexdigest()
    assert payload["randomization"]["action_library_hash"] == library_hash
    assert payload["randomization"]["learned_method_comparison_authorized"] is False
    assert payload["safety"]["maximum_commanded_joint_speed_rad_s"] < 0.09
    assert payload["safety"]["maximum_allowed_lock_error_rad"] < 0.062
    assert payload["freeze_record"]["frozen_before_first_method_trial"] is True
