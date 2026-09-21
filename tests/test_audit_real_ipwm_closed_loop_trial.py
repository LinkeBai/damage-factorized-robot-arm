import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_real_ipwm_closed_loop_trial import audit_trial, sha256


def _packet(root: Path) -> Path:
    root.mkdir()
    for name in ("daheng_FDE23080341_raw.avi", "directshow_index1_raw.avi"):
        (root / name).write_bytes(b"video")
    cycles = []
    targets = [(2000, 2100, 2200, 2300, 2400), (2000, 2100, 2205, 2295, 2400)]
    cycle_dir = root / "ipwm_replan_cycles"
    cycle_dir.mkdir()
    for i, target in enumerate(targets):
        path = cycle_dir / f"cycle_{i:02d}.npz"
        candidates = np.arange(10000 * 2 * 5, dtype=float).reshape(10000, 2, 5)
        np.savez_compressed(path, scores=np.arange(10_000),
                            candidate_references=candidates,
                            selection_eligible=np.ones(10000, dtype=bool),
                            selected_index=np.asarray(0), selected_references=candidates[0],
                            selected_first_target_raw=np.asarray(target))
        cycles.append({"cycle": i, "observation_monotonic_ns": i + 1,
                       "selected_index": 0,
                       "candidate_bank_sha256": hashlib.sha256(candidates.tobytes()).hexdigest(),
                       "candidate_count_scored": 10_000,
                       "selected_first_target_raw": list(target),
                       "cycle_evidence": str(path),
                       "cycle_evidence_sha256": sha256(path)})
    (root / "run_manifest.json").write_text(json.dumps({
        "status": "ACQUISITION_COMPLETE_UNASSESSED",
        "planner_mode": "receding_horizon", "true_closed_loop_demonstrated": True,
        "locked_joints": ["j1", "j2"], "ipwm_online_replan_cycles": cycles,
    }))
    (root / "live_task_gate.json").write_text(json.dumps({
        "goal_px": [1116.31, 570], "tolerance_px": 5,
        "gate_mode": "axiswise", "required_consecutive_frames": 3,
        "stable_goal_ever": True,
        "samples": [
            {"monotonic_ns": 1, "overhead_frame_sequence": 1, "detected": True, "current_px": [1086.31, 570]},
            {"monotonic_ns": 2, "overhead_frame_sequence": 2, "detected": True, "current_px": [1116.31, 570]},
            {"monotonic_ns": 3, "overhead_frame_sequence": 3, "detected": True, "current_px": [1116.31, 570]},
            {"monotonic_ns": 4, "overhead_frame_sequence": 4, "detected": True, "current_px": [1116.31, 570]},
        ],
    }))
    with (root / "commands.csv").open("w", newline="") as f:
        fields = ["phase"] + [f"j{i}_target_raw" for i in range(1, 6)]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for target in targets:
            writer.writerow({"phase": "fixed_trajectory",
                             **{f"j{i}_target_raw": target[i-1] for i in range(1, 6)}})
    with (root / "servo_telemetry.csv").open("w", newline="") as f:
        fields = [x for i in range(1, 6) for x in
                  (f"j{i}_position_raw", f"j{i}_target_raw")]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerow({x: 2000 + 100 * ((int(x[1])) - 1) for x in fields})
    return root


def test_accepts_complete_genuine_closed_loop_packet(tmp_path):
    result = audit_trial(_packet(tmp_path / "trial"))
    assert result["status"] == "PASS"
    assert result["task_outcome"] == "GO"
    assert result["candidate_count_each_cycle"] == [10_000, 10_000]


def test_rejects_tampered_cycle_scores(tmp_path):
    trial = _packet(tmp_path / "trial")
    with np.load(trial / "ipwm_replan_cycles/cycle_01.npz") as old:
        np.savez_compressed(trial / "ipwm_replan_cycles/cycle_01.npz",
                            scores=old["scores"] + 1,
                            selected_first_target_raw=old["selected_first_target_raw"])
    result = audit_trial(trial)
    assert result["status"] == "FAIL_CLOSED"
    assert any("sha256_mismatch" in error for error in result["errors"])


def test_three_frame_success_does_not_require_later_median_success(tmp_path):
    trial = _packet(tmp_path / "trial")
    path = trial / "live_task_gate.json"
    payload = json.loads(path.read_text())
    payload["samples"] += [
        {"monotonic_ns": i, "overhead_frame_sequence": i,
         "detected": True, "current_px": [1100, 570]} for i in range(5, 15)]
    path.write_text(json.dumps(payload))
    result = audit_trial(trial)
    assert result["endpoint_error_xy_px"][0] < -5
    assert result["task_outcome"] == "GO"


def test_duplicate_goal_frame_is_invalid(tmp_path):
    trial = _packet(tmp_path / "trial")
    path = trial / "live_task_gate.json"
    payload = json.loads(path.read_text())
    payload["samples"][-1]["overhead_frame_sequence"] = 3
    path.write_text(json.dumps(payload))
    assert audit_trial(trial)["task_outcome"] == "INVALID"


def test_aborted_acquisition_is_invalid_not_task_failure(tmp_path):
    trial = _packet(tmp_path / "trial")
    path = trial / "run_manifest.json"
    payload = json.loads(path.read_text()); payload["status"] = "ABORTED"
    path.write_text(json.dumps(payload))
    assert audit_trial(trial)["task_outcome"] == "INVALID"


def test_declared_success_does_not_replace_three_real_frames(tmp_path):
    trial = _packet(tmp_path / "trial")
    path = trial / "live_task_gate.json"
    payload = json.loads(path.read_text()); payload["samples"] = payload["samples"][:2]
    path.write_text(json.dumps(payload))
    result = audit_trial(trial)
    assert result["task_outcome"] == "INVALID"
    assert "recorded_success_disagrees_with_recomputed_three_frame_gate" in result["errors"]


def _replace_cycle(trial, change):
    path = trial / "ipwm_replan_cycles/cycle_00.npz"
    with np.load(path) as old:
        arrays = {k: old[k] for k in old.files}
    change(arrays)
    np.savez_compressed(path, **arrays)
    manifest = trial / "run_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["ipwm_online_replan_cycles"][0]["cycle_evidence_sha256"] = sha256(path)
    manifest.write_text(json.dumps(payload))


def test_scores_alone_cannot_establish_full_candidate_archival(tmp_path):
    trial = _packet(tmp_path / "trial")
    _replace_cycle(trial, lambda z: z.pop("candidate_references"))
    result = audit_trial(trial)
    assert result["task_outcome"] == "INVALID"
    assert any(e.startswith("missing_candidate_archive_fields:") for e in result["errors"])


def test_selected_trajectory_must_be_from_scored_bank(tmp_path):
    trial = _packet(tmp_path / "trial")
    _replace_cycle(trial, lambda z: z.update(selected_references=z["selected_references"] + 1))
    result = audit_trial(trial)
    assert result["task_outcome"] == "INVALID"
    assert any(e.startswith("selected_references_not_from_candidate_bank:") for e in result["errors"])


def test_integer_eligibility_mask_is_rejected(tmp_path):
    trial = _packet(tmp_path / "trial")
    _replace_cycle(trial, lambda z: z.update(selection_eligible=z["selection_eligible"].astype(int)))
    assert audit_trial(trial)["task_outcome"] == "INVALID"


def test_final_plan_cancellation_requires_success_during_inference():
    from scripts.audit_real_ipwm_closed_loop_trial import cancelled_final_plan_after_success
    cycles = [{"cycle": i} for i in range(3)]
    cycles[-1].update(observation_monotonic_ns=100, inference_s=1e-6)
    run = {"live_goal_early_stop": {"triggered": True, "reason": "online_receding_horizon_goal_guard"}}
    gate = {"goal_px": [0, 0], "samples": [
        {"monotonic_ns": t, "overhead_frame_sequence": i, "detected": True, "current_px": [0, 0]}
        for i, t in enumerate([110, 120, 130])]}
    commands = [{"phase": "fixed_trajectory", "dispatch_monotonic_ns": 50+i, "segment_index": i}
                for i in range(2)]
    check = lambda: cancelled_final_plan_after_success(cycles[-1], cycles, run, gate, commands)
    assert check()
    commands[-1]["dispatch_monotonic_ns"] = 140
    assert not check()
    commands[-1]["dispatch_monotonic_ns"] = 51
    gate["samples"][-1]["overhead_frame_sequence"] = 1
    assert not check()
    gate["samples"][-1]["overhead_frame_sequence"] = 2
    gate["samples"][-1]["monotonic_ns"] = 2000
    assert not check()
    cycles[-1].update(inference_started_monotonic_ns=105,
                      inference_completed_monotonic_ns=1105,
                      archive_completed_monotonic_ns=2100)
    assert check()
    cycles[-1]["archive_completed_monotonic_ns"] = 1000
    assert not check()
    cycles[-1].pop("archive_completed_monotonic_ns")
    assert not check()
