import hashlib
import json
from pathlib import Path

from scripts.audit_real_ipwm_trial import audit


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path: Path) -> Path:
    files = {}
    for name in ("checkpoint", "model_config", "candidate_archive", "selected_trajectory", "action_bridge", "simulation_model"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        files[name] = {"path": path.name, "sha256": _hash(path)}
    run = tmp_path / "run_manifest.json"
    run.write_text(json.dumps({
        "trial_id": "trial-1", "condition": "D3",
        "trajectory_sha256": files["selected_trajectory"]["sha256"],
    }), encoding="utf-8")
    files["run_manifest"] = {"path": run.name, "sha256": _hash(run)}
    packet = {
        "schema_version": "real_ipwm_decision_v1", "trial_id": "trial-1",
        "method": "selective_ipwm", "planner_mode": "open_loop_sequence", "condition": "D3",
        "files": files,
        "initial_observation": {
            "joint_position_rad": [0] * 5, "joint_velocity_rad_s": [0] * 5,
            "object_xy_m": [0.2, 0.1], "goal_xy_m": [0.23, 0.1],
            "telemetry_timestamp_utc": "2026-09-03T00:00:00Z",
            "task_plane_calibration_sha256": "abc",
        },
        "decision": {
            "candidate_count": 2, "selected_candidate_index": 0,
            "predicted_scores": [0.01, 0.02],
            "selection_rule": "minimum_predicted_terminal_object_to_goal_distance",
        },
        "bridge": {"simulation_model": "genkiarm_push.xml", "safety_audit": "PASS", "clipped_action_count": 0},
    }
    path = tmp_path / "model_decision_manifest.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def test_complete_packet_passes(tmp_path):
    result = audit(_packet(tmp_path))
    assert result["status"] == "PASS"
    assert result["is_model_in_loop_evidence"] is True


def test_wrong_method_and_tampering_fail_closed(tmp_path):
    path = _packet(tmp_path)
    packet = json.loads(path.read_text())
    packet["method"] = "fixed_trajectory"
    path.write_text(json.dumps(packet))
    (tmp_path / "checkpoint.bin").write_bytes(b"tampered")
    result = audit(path)
    assert result["status"] == "FAIL_CLOSED"
    assert result["is_model_in_loop_evidence"] is False
    assert any("method" in error for error in result["errors"])
    assert any("checkpoint sha256" in error for error in result["errors"])
