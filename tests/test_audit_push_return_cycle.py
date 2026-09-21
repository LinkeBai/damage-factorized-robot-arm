import json

from scripts.audit_push_return_cycle import REQUIRED_FILES, audit_cycle


def make_packet(tmp_path, duration=9.8):
    manifest = {
        "terminal_state": "COMPLETE",
        "motion_duration_s": duration,
        "start_gate_passed": True,
        "forward_goal_reached": True,
        "reset_goal_reached": True,
        "home_gate_passed": True,
        "scene_clear_gate_passed": True,
        "camera_gate_passed": True,
        "servo_feedback_gate_passed": True,
        "electrical_gate_passed": True,
        "raw_sources_immutable": True,
    }
    for name in REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"evidence\n")
    (tmp_path / "cycle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_complete_packet_under_ten_seconds_passes(tmp_path):
    result = audit_cycle(make_packet(tmp_path))
    assert result["status"] == "PASS"
    assert len(result["artifacts"]) == len(REQUIRED_FILES)


def test_slow_cycle_fails(tmp_path):
    result = audit_cycle(make_packet(tmp_path, duration=10.01))
    assert result["status"] == "FAIL"
    assert any("exceeds allowed" in error for error in result["errors"])


def test_forward_only_cannot_pass(tmp_path):
    make_packet(tmp_path)
    path = tmp_path / "cycle_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["reset_goal_reached"] = False
    path.write_text(json.dumps(manifest))
    assert audit_cycle(tmp_path)["status"] == "FAIL"


def test_missing_raw_video_fails(tmp_path):
    make_packet(tmp_path)
    (tmp_path / "directshow_index1_raw.avi").unlink()
    assert audit_cycle(tmp_path)["status"] == "FAIL"

