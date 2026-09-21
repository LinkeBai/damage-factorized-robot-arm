import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_real_robot_trial_packet import audit_trial_packet, main


DAHENG_CAMERA = "daheng_sn_TEST123"
DIRECTSHOW_CAMERA = "directshow_index_1"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), 30 + index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _normal_packet(tmp_path: Path, trial_id: str = "trial-001") -> Path:
    trial_dir = tmp_path / trial_id
    trial_dir.mkdir()
    daheng_name = "daheng_TEST123_raw.avi"
    directshow_name = "directshow_index1_raw.avi"
    _write_video(trial_dir / daheng_name, 2)
    _write_video(trial_dir / directshow_name, 2)

    command_joint_fields = [f"j{index}_target_raw" for index in range(1, 6)]
    _write_csv(
        trial_dir / "commands.csv",
        ["command_index", *command_joint_fields],
        [{"command_index": 0, **{field: 2000 for field in command_joint_fields}}],
    )
    telemetry_joint_fields = [
        field
        for index in range(1, 6)
        for field in (f"j{index}_position_raw", f"j{index}_target_raw")
    ]
    _write_csv(
        trial_dir / "servo_telemetry.csv",
        ["sample_index", *telemetry_joint_fields],
        [{"sample_index": 0, **{field: 2000 for field in telemetry_joint_fields}}],
    )
    _write_csv(
        trial_dir / "frame_timestamps.csv",
        ["camera", "frame_index", "capture_mid_monotonic_ns"],
        [
            {"camera": DAHENG_CAMERA, "frame_index": index,
             "capture_mid_monotonic_ns": 100 + index}
            for index in range(2)
        ] + [
            {"camera": DIRECTSHOW_CAMERA, "frame_index": index,
             "capture_mid_monotonic_ns": 200 + index}
            for index in range(2)
        ],
    )
    manifest = {
        "status": "ACQUISITION_COMPLETE_UNASSESSED",
        "trial_id": trial_id,
        "trajectory_id": "frozen_intact_trajectory",
        "condition": "intact",
        "locked_joint": None,
        "locked_target_raw": None,
        "waypoint_sha256": "1" * 64,
        "camera_devices": {"daheng_serial": "TEST123", "directshow_index": 1},
        "camera_frame_counts": {"daheng": 2, "directshow": 2},
        "command_rows": 1,
        "telemetry_rows": 1,
        "artifacts": {
            "daheng_video": daheng_name,
            "directshow_video": directshow_name,
            "commands": "commands.csv",
            "servo_telemetry": "servo_telemetry.csv",
            "frame_timestamps": "frame_timestamps.csv",
        },
    }
    (trial_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return trial_dir


def test_complete_normal_packet_passes_with_hashes_and_decoded_counts(
    tmp_path: Path,
) -> None:
    trial_dir = _normal_packet(tmp_path)

    result = audit_trial_packet(
        trial_dir,
        expected_trial_id="trial-001",
        expected_condition="intact",
        expected_trajectory_id="frozen_intact_trajectory",
    )

    assert result["status"] == "PASS"
    assert result["packet_integrity_status"] == "PASS"
    assert result["trial_validity_status"] == "VALID_UNASSESSED"
    assert result["valid_trial"] is True
    assert result["artifact_count"] == 6
    assert {row["role"] for row in result["artifacts"]} == {
        "run_manifest", "daheng_video", "directshow_video", "commands",
        "servo_telemetry", "frame_timestamps",
    }
    for artifact in result["artifacts"]:
        assert artifact["bytes"] > 0
        assert len(artifact["sha256"]) == 64
    videos = [row for row in result["artifacts"] if row["role"].endswith("video")]
    assert [row["decoded_frame_count"] for row in videos] == [2, 2]


def test_abort_is_retained_and_hashed_but_never_counted_as_valid(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial-abort-001"
    trial_dir.mkdir()
    (trial_dir / "daheng_TEST123_raw.avi").write_bytes(b"partial-aborted-video")
    (trial_dir / "commands.csv").write_text("command_index\n", encoding="utf-8")
    (trial_dir / "run_manifest.json").write_text(json.dumps({
        "status": "ABORTED_TORQUE_OFF_REQUESTED",
        "trial_id": trial_dir.name,
        "trajectory_id": "frozen_intact_trajectory",
        "condition": "intact",
        "locked_joint": None,
        "locked_target_raw": None,
        "aborted_utc": "2026-09-02T00:00:00+00:00",
        "failure_type": "RuntimeError",
        "failure_message": "camera stopped",
    }), encoding="utf-8")

    result = audit_trial_packet(trial_dir)

    assert result["status"] == "RETAINED_ABORT"
    assert result["packet_integrity_status"] == "PASS"
    assert result["trial_validity_status"] == "INVALID_ABORTED"
    assert result["valid_trial"] is False
    assert sum(int(item["valid_trial"]) for item in [result]) == 0
    assert {row["role"] for row in result["artifacts"]} == {
        "run_manifest", "daheng_video", "commands",
    }
    assert all(len(row["sha256"]) == 64 for row in result["artifacts"])
    # A retained abort is not a successful valid-trial CLI gate.
    assert main([str(trial_dir)]) == 2


def test_normal_packet_fails_for_missing_joint_field_and_frame_mismatch(
    tmp_path: Path,
) -> None:
    trial_dir = _normal_packet(tmp_path)
    telemetry = trial_dir / "servo_telemetry.csv"
    with telemetry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    fields.remove("j5_position_raw")
    _write_csv(telemetry, fields, [{field: rows[0][field] for field in fields}])
    manifest_path = trial_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["camera_frame_counts"]["daheng"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_trial_packet(trial_dir)

    assert result["status"] == "FAIL"
    assert result["packet_integrity_status"] == "FAIL"
    assert result["valid_trial"] is False
    assert any("j5_position_raw" in error for error in result["errors"])
    assert any("daheng video frame count mismatch" in error for error in result["errors"])
    assert any("frame_timestamps count mismatch" in error for error in result["errors"])


def test_manifest_identity_must_match_trial_directory_and_expected_schedule(
    tmp_path: Path,
) -> None:
    trial_dir = _normal_packet(tmp_path)

    result = audit_trial_packet(
        trial_dir,
        expected_trial_id="trial-999",
        expected_condition="D2",
        expected_trajectory_id="another_trajectory",
    )

    assert result["status"] == "FAIL"
    assert result["valid_trial"] is False
    assert any("does not match expected" in error for error in result["errors"])
