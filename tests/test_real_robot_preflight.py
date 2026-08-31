from pathlib import Path

import yaml

from scripts.audit_real_robot_preflight import audit


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data/real_robot/push_schedule_seed20260901.csv"


def filled_manifest(tmp_path: Path) -> Path:
    calibration_left = tmp_path / "left.yaml"
    calibration_horizontal = tmp_path / "horizontal.yaml"
    sync_video = tmp_path / "sync.mp4"
    bridge = tmp_path / "action_bridge.yaml"
    validation = tmp_path / "action_validation.json"
    for path in (calibration_left, calibration_horizontal, sync_video, bridge, validation):
        path.write_text("x", encoding="utf-8")
    directories = [tmp_path / name for name in ("left", "horizontal", "logs", "backup1", "backup2")]
    for path in directories:
        path.mkdir()
    payload = yaml.safe_load((ROOT / "data/real_robot/session_manifest_template.yaml").read_text(encoding="utf-8"))
    payload.update({"session_id": "S1", "date_local": "2026-09-01", "operator": "AB"})
    payload["hardware"].update({
        "robot_asset_id": "R1", "gripper_asset_id": "G1", "block_asset_id": "B1",
        "emergency_stop_checked": True, "joint_direction_check_complete": True,
        "low_speed_stop_check_complete": True,
    })
    payload["cameras"].update({
        "left_eye_to_hand_serial": "L", "horizontal_eye_to_hand_serial": "H",
        "left_calibration_file": str(calibration_left),
        "horizontal_calibration_file": str(calibration_horizontal),
        "synchronization_event_video": str(sync_video),
    })
    payload["safety"].update({
        "maximum_commanded_joint_speed_rad_s": 0.0873,
        "maximum_allowed_lock_error_rad": 0.0611,
        "workspace_boundary_description": "marked rectangle",
    })
    payload["randomization"].update({
        "schedule_file": str(SCHEDULE),
        "schedule_sha256_before_trials": "79139bca3b61866643e00ef35d724cdd4185fb14a8f115faa942635f27f4510d",
        "physical_reset_fixture_description": "three marked positions",
        "action_library_hash": "abc",
        "action_interface_bridge_file": str(bridge),
        "action_interface_validation_file": str(validation),
        "learned_method_comparison_authorized": True,
    })
    payload["data_roots"].update(dict(zip(
        ("left_video_directory", "horizontal_video_directory", "control_log_directory", "backup_copy_1", "backup_copy_2"),
        map(str, directories))))
    payload["freeze_record"].update({
        "frozen_before_first_method_trial": True,
        "freeze_timestamp_local": "2026-09-01T08:00:00+08:00",
        "operator_signature_or_initials": "AB",
    })
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_filled_preflight_passes(tmp_path: Path) -> None:
    result = audit(filled_manifest(tmp_path), SCHEDULE)
    assert result["status"] == "PASS"
    assert result["authorization"] == "FORMAL_TRIALS_MAY_START"
    assert result["schedule_trials"] == 50
    assert result["schedule_pairs"] == 25


def test_template_preflight_fails_closed() -> None:
    result = audit(ROOT / "data/real_robot/session_manifest_template.yaml", SCHEDULE, False)
    assert result["status"] == "FAIL"
    assert result["authorization"] == "DO_NOT_START_FORMAL_TRIALS"
    assert any("session_id" in error for error in result["errors"])
