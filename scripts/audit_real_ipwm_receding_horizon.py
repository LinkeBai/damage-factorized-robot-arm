"""Hardware-free preflight for the genuine real-IPWM receding-horizon path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from robotarm.deployment.fixed_raw_trajectory import load_safety_envelope, locked_indices_for_condition
from robotarm.deployment.ipwm_receding_horizon import IPWMRecedingHorizonPlanner
from scripts.prepare_real_ipwm_trial import load_model, score_references_terminal_only


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs/icra_confirmation_d3_query_selective_w10/seed27/model.pt")
    ap.add_argument("--model-config", type=Path, default=ROOT / "config/experiment/icra_primary_d2d4_eval_strict_3seed_v1.yaml")
    ap.add_argument("--axis-calibration", type=Path, default=ROOT / "results/real_robot/push_axis_current_epoch_20260903.json")
    ap.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--execution-reference-index", type=int, default=0)
    ap.add_argument("--task-start-px", type=float, nargs=2, required=True)
    ap.add_argument("--task-goal-px", type=float, nargs=2, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real closed-loop latency preflight")
    archive = np.load(args.archive, allow_pickle=False)
    refs = np.asarray(archive["q_reference_rad"], dtype=float)
    if len(refs) < 10_000:
        raise SystemExit("archive has fewer than 10,000 candidates")
    indices = np.linspace(refs.shape[1] / args.horizon - 1, refs.shape[1] - 1,
                          args.horizon).round().astype(int)
    initial = np.asarray(archive["initial_state"], dtype=float)
    deltas = refs[:, indices] - initial[None, None, :5]
    model = load_model(args.checkpoint, args.model_config, torch.device("cuda"))
    def score(state, candidates, goal, mask, angle):
        return score_references_terminal_only(model, state, candidates, goal, mask, angle, 10_000)
    axis = json.loads(args.axis_calibration.read_text(encoding="utf-8"))["diagnostics"]
    safety = load_safety_envelope(args.safety)
    ranges = np.deg2rad(np.asarray([[j.min_deg, j.max_deg] for j in safety.joints]))
    planner = IPWMRecedingHorizonPlanner(
        deltas, task_start_px=tuple(args.task_start_px), task_goal_px=tuple(args.task_goal_px),
        locked_indices=locked_indices_for_condition(args.condition),
        score_function=score, metres_per_pixel=float(axis["metres_per_pixel"]),
        base_xy_per_pixel=np.asarray(axis["base_xy_per_pixel"]),
        base_xy_intercept_m=np.asarray(axis["base_xy_intercept_m"]),
        joint_ranges_rad=ranges,
        base_eligible=np.asarray(archive["selection_eligible"], dtype=bool),
        execution_reference_index=args.execution_reference_index,
    )
    q0 = initial[:5]
    lock = np.asarray(archive["lock_angle"], dtype=float)
    midpoint = tuple((np.asarray(args.task_start_px) + np.asarray(args.task_goal_px)) / 2)
    first = planner.plan(joint_q=q0, joint_qd=np.zeros(5), object_px=tuple(args.task_start_px),
                         observation_monotonic_ns=1, lock_angles=lock)
    q1 = first.selected_first_reference
    second = planner.plan(joint_q=q1, joint_qd=q1-q0, object_px=midpoint,
                          observation_monotonic_ns=2, lock_angles=lock)
    counterfactual_stale = planner.plan(
        joint_q=q1, joint_qd=q1-q0, object_px=tuple(args.task_start_px),
        observation_monotonic_ns=3, lock_angles=lock,
    )
    locked = locked_indices_for_condition(args.condition)
    locked_ok = all(np.allclose(first.selected_references[:, j], lock[j]) and
                    np.allclose(second.selected_references[:, j], lock[j]) for j in locked)
    command_changed = not np.allclose(first.selected_first_reference,
                                      second.selected_first_reference, atol=1e-8)
    observation_causes_change = not np.allclose(
        second.selected_first_reference,
        counterfactual_stale.selected_first_reference,
        atol=1e-8,
    )
    payload = {
        "status": "PASS" if locked_ok and command_changed and observation_causes_change else "FAIL",
        "semantics": "hardware-free deployment preflight; not a physical result",
        "condition": args.condition,
        "task_start_px": args.task_start_px,
        "task_goal_px": args.task_goal_px,
        "candidate_count_each_cycle": len(first.scores),
        "horizon": args.horizon,
        "execution_reference_index": args.execution_reference_index,
        "cycle_0": {"object_px": first.observation_px, "selected_index": first.selected_index,
                    "inference_s": first.inference_s,
                    "first_reference_rad": first.selected_first_reference.tolist()},
        "cycle_1": {"object_px": second.observation_px, "selected_index": second.selected_index,
                    "inference_s": second.inference_s,
                    "first_reference_rad": second.selected_first_reference.tolist()},
        "second_command_changed_after_new_observation": command_changed,
        "same_joint_state_counterfactual_observation_changes_command": observation_causes_change,
        "locked_coordinates_invariant": locked_ok,
        "maximum_inference_s": max(first.inference_s, second.inference_s),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
