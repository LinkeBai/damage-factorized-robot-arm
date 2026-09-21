"""Fail-closed audit and outcome adjudication for a physical IPWM loop."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cancelled_final_plan_after_success(cycle, cycles, run, gate, commands):
    """Accept a final plan cancelled by the asynchronous image goal guard.

    The camera thread may establish the three-frame gate after inference has
    selected the next short reference.  Depending on timing, the executor may
    cancel that reference before its first microstep or during its exact
    monotone prefix.  Requiring the selected endpoint in either case would
    incorrectly turn a successful stop into an invalid packet.
    """
    if not cycles or cycle is not cycles[-1] or len(cycles) < 3:
        return False
    stop = run.get("live_goal_early_stop", {})
    if (stop.get("triggered") is not True
            or stop.get("reason") != "online_receding_horizon_goal_guard"):
        return False
    try:
        observed = int(cycle["observation_monotonic_ns"])
        duration = float(cycle["inference_s"])
        if not np.isfinite(duration) or duration <= 0:
            return False
        plan_ready = observed + duration * 1e9
        boundaries = ("inference_started_monotonic_ns", "inference_completed_monotonic_ns",
                      "archive_completed_monotonic_ns")
        if any(key in cycle for key in boundaries):
            started, completed, archived = (int(cycle[key]) for key in boundaries)
            if not observed <= started <= completed <= archived:
                return False
            plan_ready = archived
        motion = [r for r in commands if r.get("phase") == "fixed_trajectory"]
        if not motion:
            return False
        later = [r for r in motion if int(r["dispatch_monotonic_ns"]) >= observed]
        cycle_index = int(cycle["cycle"])
        if any(int(r["segment_index"]) != cycle_index for r in later):
            return False
        segments = {int(r["segment_index"]) for r in motion}
        if not set(int(c["cycle"]) for c in cycles[:-1]).issubset(segments):
            return False
        # A partially dispatched final short block must be an axiswise
        # monotone prefix ending within two raw ticks of the selected target.
        # No command may be dispatched after the success timestamp below.
        target = np.asarray(cycle["selected_first_target_raw"], dtype=int)
        start_raw = np.asarray(cycle["joint_raw"], dtype=int)
        if later:
            dispatched = np.asarray([
                [int(r[f"j{i}_target_raw"]) for i in range(1, 6)] for r in later
            ], dtype=int)
            lo = np.minimum(start_raw, target)
            hi = np.maximum(start_raw, target)
            if (np.any(dispatched < lo[None, :]) or np.any(dispatched > hi[None, :])
                    or np.any(np.diff(dispatched, axis=0) * np.sign(target - start_raw)[None, :] < 0)
                    or np.max(np.abs(dispatched[-1] - target)) > 2):
                return False
        count = 0
        prior_time = prior_frame = -1
        for sample in gate.get("samples", []):
            timestamp = int(sample["monotonic_ns"])
            frame = int(sample["overhead_frame_sequence"])
            if timestamp <= prior_time or frame <= prior_frame:
                return False
            prior_time, prior_frame = timestamp, frame
            point = np.asarray(sample.get("current_px", [np.nan, np.nan]), dtype=float)
            inside = (sample.get("detected") and point.shape == (2,)
                      and np.all(np.isfinite(point))
                      and np.max(np.abs(point - np.asarray(gate["goal_px"]))) <= 5)
            count = count + 1 if inside else 0
            if count >= 3 and timestamp >= observed:
                if any(int(r["dispatch_monotonic_ns"]) > timestamp for r in later):
                    return False
                # Legacy packets can establish the asynchronous gate while
                # inference/archive is still running and therefore have no
                # post-observation dispatch.  Newer packets may establish it
                # after planning, either before dispatch or at the end of a
                # monotone microstep prefix.  Both are the same frozen
                # stop-on-go protocol; do not require fields introduced later.
                if later and timestamp < plan_ready:
                    return False
                return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return False


def audit_trial(trial: Path) -> dict:
    trial = trial.resolve()
    errors: list[str] = []
    run_path = trial / "run_manifest.json"
    gate_path = trial / "live_task_gate.json"
    telemetry_path = trial / "servo_telemetry.csv"
    commands_path = trial / "commands.csv"
    required = [run_path, gate_path, telemetry_path, commands_path,
                trial / "daheng_FDE23080341_raw.avi",
                trial / "directshow_index1_raw.avi"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{path.name}")
    if errors:
        return {"status": "FAIL_CLOSED", "errors": errors, "trial": str(trial)}
    run = json.loads(run_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if run.get("status") != "ACQUISITION_COMPLETE_UNASSESSED":
        errors.append("acquisition_not_complete")
    if (gate.get("gate_mode") != "axiswise"
            or gate.get("tolerance_px") != 5.0
            or gate.get("required_consecutive_frames") != 3):
        errors.append("success_gate_not_frozen_5px_three_frames")
    cycles = run.get("ipwm_online_replan_cycles", [])
    if run.get("planner_mode") != "receding_horizon":
        errors.append("planner_mode_not_receding_horizon")
    if run.get("true_closed_loop_demonstrated") is not True or len(cycles) < 2:
        errors.append("fewer_than_two_proven_observation_plan_action_cycles")
    observation_times = [int(c.get("observation_monotonic_ns", -1)) for c in cycles]
    if observation_times != sorted(set(observation_times)):
        errors.append("observation_timestamps_not_strictly_increasing")
    if any(int(c.get("candidate_count_scored", 0)) < 10_000 for c in cycles):
        errors.append("candidate_budget_below_10000")

    with commands_path.open(newline="", encoding="utf-8") as f:
        command_rows = list(csv.DictReader(f))
    fixed_targets = {
        tuple(int(row[f"j{i}_target_raw"]) for i in range(1, 6))
        for row in command_rows if row.get("phase") == "fixed_trajectory"
    }
    for cycle in cycles:
        npz_path = Path(cycle.get("cycle_evidence", ""))
        if not npz_path.is_file():
            errors.append(f"missing_cycle_npz:{cycle.get('cycle')}")
            continue
        if (cycle.get("cycle_evidence_sha256") is not None
                and sha256(npz_path) != cycle.get("cycle_evidence_sha256")):
            errors.append(f"cycle_npz_sha256_mismatch:{cycle.get('cycle')}")
        evidence = np.load(npz_path, allow_pickle=False)
        if evidence["scores"].shape != (int(cycle["candidate_count_scored"]),):
            errors.append(f"score_shape_mismatch:{cycle.get('cycle')}")
        if not np.all(np.isfinite(evidence["scores"])):
            errors.append(f"nonfinite_candidate_scores:{cycle.get('cycle')}")
        candidate_fields = {"candidate_references", "selection_eligible", "selected_index",
                            "selected_references"}
        missing_candidate_fields = candidate_fields - set(evidence.files)
        if missing_candidate_fields:
            errors.append(f"missing_candidate_archive_fields:{cycle.get('cycle')}:"
                          + ",".join(sorted(missing_candidate_fields)))
        else:
            candidates = evidence["candidate_references"]
            digest = hashlib.sha256(np.ascontiguousarray(candidates).tobytes()).hexdigest()
            if digest != cycle.get("candidate_bank_sha256"):
                errors.append(f"candidate_bank_sha256_mismatch:{cycle.get('cycle')}")
            eligible = evidence["selection_eligible"]
            selected_array = evidence["selected_index"]
            selected = int(selected_array) if selected_array.shape == () and np.issubdtype(selected_array.dtype, np.integer) else -1
            bank_valid = (candidates.ndim == 3
                          and candidates.shape[0] == int(cycle["candidate_count_scored"])
                          and candidates.shape[1] > 0 and candidates.shape[2] == 5
                          and np.all(np.isfinite(candidates)))
            if not bank_valid:
                errors.append(f"invalid_candidate_bank_shape_or_values:{cycle.get('cycle')}")
            if (eligible.dtype != np.bool_ or eligible.shape != evidence["scores"].shape
                    or not np.any(eligible) or evidence["scores"].ndim != 1
                    or selected != int(np.argmin(np.where(eligible, evidence["scores"], np.inf)))):
                errors.append(f"selected_candidate_not_eligible_argmin:{cycle.get('cycle')}")
            if (not bank_valid or not 0 <= selected < len(candidates)
                    or not np.array_equal(evidence["selected_references"], candidates[selected])):
                errors.append(f"selected_references_not_from_candidate_bank:{cycle.get('cycle')}")
            if selected != cycle.get("selected_index"):
                errors.append(f"selected_index_manifest_mismatch:{cycle.get('cycle')}")
        target = tuple(int(v) for v in evidence["selected_first_target_raw"])
        if target != tuple(int(v) for v in cycle["selected_first_target_raw"]):
            errors.append(f"cycle_target_manifest_mismatch:{cycle.get('cycle')}")
        if target not in fixed_targets and not cancelled_final_plan_after_success(
                cycle, cycles, run, gate, command_rows):
            errors.append(f"selected_target_not_executed:{cycle.get('cycle')}")

    with telemetry_path.open(newline="", encoding="utf-8") as f:
        telemetry = list(csv.DictReader(f))
    locked = [int(name[1:]) for name in run.get("locked_joints", [])]
    lock_drift = {}
    for joint in locked:
        values = [abs(int(row[f"j{joint}_position_raw"]) -
                      int(row[f"j{joint}_target_raw"])) for row in telemetry]
        lock_drift[f"j{joint}_maximum_feedback_to_target_ticks"] = max(values, default=0)

    detected = [sample for sample in gate.get("samples", []) if sample.get("detected")]
    if not detected:
        errors.append("no_reliable_cube_observation")
        endpoint_error = [None, None]
        displacement = None
    else:
        start = np.asarray(detected[0]["current_px"], dtype=float)
        end = np.median(np.asarray([s["current_px"] for s in detected[-min(10, len(detected)):]],
                                   dtype=float), axis=0)
        goal = np.asarray(gate["goal_px"], dtype=float)
        endpoint_error = (end - goal).tolist()
        displacement = float(np.linalg.norm(end - start))
    tolerance = float(gate.get("tolerance_px", 5.0))
    consecutive = maximum_consecutive = 0
    prior_time = prior_frame = -1
    for sample in gate.get("samples", []):
        timestamp = int(sample.get("monotonic_ns", -1))
        frame = int(sample.get("overhead_frame_sequence", -1))
        if timestamp <= prior_time or frame <= prior_frame:
            errors.append("goal_frames_missing_or_not_strictly_increasing")
        prior_time, prior_frame = timestamp, frame
        point = np.asarray(sample.get("current_px", [np.nan, np.nan]), dtype=float)
        inside = (sample.get("detected") and point.shape == (2,)
                  and np.all(np.isfinite(point))
                  and np.max(np.abs(point - np.asarray(gate["goal_px"]))) <= 5.0)
        consecutive = consecutive + 1 if inside else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
    if bool(gate.get("stable_goal_ever")) != (maximum_consecutive >= 3):
        errors.append("recorded_success_disagrees_with_recomputed_three_frame_gate")
    success = bool(not errors and maximum_consecutive >= 3)
    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "is_genuine_physical_ipwm_closed_loop_evidence": not errors,
        "task_outcome": "INVALID" if errors else ("GO" if success else "NO_GO"),
        "success": success,
        "criterion": "at least 3 consecutive distinct reliable frames with each-axis error <= 5px",
        "adjudication_version": "three_distinct_frames_v2",
        "endpoint_error_role": "last-ten-frame median diagnostic only; not an additional success gate",
        "endpoint_error_xy_px": endpoint_error,
        "observed_displacement_px": displacement,
        "recomputed_maximum_consecutive_goal_frames": maximum_consecutive,
        "replan_cycles": len(cycles),
        "candidate_count_each_cycle": [c.get("candidate_count_scored") for c in cycles],
        "locked_joint_drift": lock_drift,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in required},
        "errors": errors,
        "trial": str(trial),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trial", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = audit_trial(args.trial)
    output = args.output or args.trial / "ipwm_closed_loop_evidence.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
