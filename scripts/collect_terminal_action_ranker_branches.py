"""Collect full-episode carrier-resumed labels for contact action sequences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.training.controllers import joint_reference_action
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.diagnose_ipwm_action_ranking import build_models, predicted_costs, spearman
from scripts.diagnose_ipwm_carrier_policy_ranking import has_contact, initialize_env
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_ipwm_closed_loop_audit import plan_push
from scripts.run_push_benchmark import collect_push_domains


def carrier_proposal(
    carrier, observation, nominal, target, mask, angle, env, locked,
    *, candidates, horizon, iterations, seed, guard_threshold,
):
    action = plan_push(
        carrier, observation["state"], nominal, target[:2], mask, angle,
        env.joint_ranges, locked, candidates=candidates, horizon=horizon,
        iterations=iterations, seed=seed,
    )
    return nominal if np.linalg.norm(action - nominal) > guard_threshold else action


def replay_prefix(domain, q0a, target, locked, approach_steps, history):
    env, observation, push_reference = initialize_env(
        domain, q0a, target, locked, approach_steps
    )
    for action in history:
        observation = env.step(action)["observation"]
    return env, observation, push_reference


def terminal_rollout(
    domain, q0a, target, locked, approach_steps, history, candidate_sequence,
    carrier, mask, angle, *, target_index, total_push_steps, planner_candidates,
    planner_horizon, planner_iterations, guard_threshold, seed,
):
    env, observation, push_reference = replay_prefix(
        domain, q0a, target, locked, approach_steps, history
    )
    contacts = fallbacks = 0
    for action in candidate_sequence:
        observation = env.step(action)["observation"]
        contacts += int(has_contact(env))
    start = len(history) + len(candidate_sequence)
    for step in range(start, total_push_steps):
        nominal = joint_reference_action(
            observation["state"][:10], push_reference, locked_joints=locked
        )
        action = plan_push(
            carrier, observation["state"], nominal, target[:2], mask, angle,
            env.joint_ranges, locked, candidates=planner_candidates,
            horizon=planner_horizon, iterations=planner_iterations,
            seed=seed * 10_000 + target_index * 100 + step,
        )
        if np.linalg.norm(action - nominal) > guard_threshold:
            action = nominal
            fallbacks += 1
        observation = env.step(action)["observation"]
        contacts += int(has_contact(env))
    return {
        "final_cost_m": float(np.linalg.norm(env.block_pos() - target[:2])),
        "contact_steps_after_trigger": contacts,
        "fallback_steps_after_trigger": fallbacks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--domain", default="D3__high_damping")
    parser.add_argument("--target-split", choices=("calibration", "validation", "evaluation"), required=True)
    parser.add_argument("--target-indices", nargs="+", type=int, required=True)
    parser.add_argument("--approach-steps", type=int, default=60)
    parser.add_argument("--total-push-steps", type=int, default=90)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--planner-candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--guard-threshold", type=float, default=0.85)
    parser.add_argument(
        "--base-sequence-mode",
        choices=("one_shot_cem", "receding_oracle_audit"),
        default="one_shot_cem",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    domains = {d.domain_id: d for d in (*protocol.train, *protocol.validation, *protocol.test)}
    domain = domains[args.domain]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, carrier, carrier_wrapped, selective, encoder = build_models(
        cfg, args.seed, args.model, device
    )
    domain_index = list(domains).index(args.domain)
    calibration_seed = args.seed * 100_000 + domain_index * 1000 + 100
    key = json.dumps({"kind": "push_context_calibration", "seed": calibration_seed,
                      "domain": args.domain, "budget": 25, "q0a": q0a}, sort_keys=True)
    calibration = cached_collect(
        args.cache_dir, key,
        lambda: collect_push_domains(
            (domain,), trajectories_per_domain=1, steps=25, seed=calibration_seed,
            targets=tuple(x.as_array() for x in targets.calibration), excitation="active",
            block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
            goal_exploration_std=float(q0a["goal_exploration_std"]),
        ),
    )[0]
    mask, angle = _damage_tensors([domain.damage], device)
    with torch.no_grad():
        context, _ = encoder(
            calibration.states[None].to(device), calibration.actions[None].to(device),
            mask, return_uncertainty=True,
        )
    context = context[0] * float(cfg.get("context_posterior_scale", 1.0))
    full.set_intervention_context(context)
    carrier.set_intervention_context(context)
    selective.set_residual_context(context)

    locked = tuple(domain.damage.locked)
    pool = getattr(targets, args.target_split)
    rows, summaries = [], []
    for target_index in args.target_indices:
        item, target = pool[target_index], pool[target_index].as_array()
        env, observation, push_reference = initialize_env(
            domain, q0a, target, locked, args.approach_steps
        )
        history = []
        for step in range(args.total_push_steps):
            if has_contact(env):
                break
            nominal = joint_reference_action(
                observation["state"][:10], push_reference, locked_joints=locked
            )
            action = carrier_proposal(
                carrier_wrapped, observation, nominal, target, mask, angle, env, locked,
                candidates=args.planner_candidates, horizon=args.horizon,
                iterations=args.iterations,
                seed=args.seed * 10_000 + target_index * 100 + step,
                guard_threshold=args.guard_threshold,
            )
            history.append(action.copy())
            observation = env.step(action)["observation"]
        if not has_contact(env):
            raise RuntimeError(f"carrier policy did not contact for {item.target_id}")
        state = observation["state"].copy()

        if args.base_sequence_mode == "one_shot_cem":
            nominal = joint_reference_action(
                state[:10], push_reference, locked_joints=locked
            )
            carrier_sequence = plan_push(
                carrier_wrapped, state, nominal, target[:2], mask, angle,
                env.joint_ranges, locked, candidates=args.planner_candidates,
                horizon=args.horizon, iterations=args.iterations,
                seed=args.seed * 10_000 + target_index * 100 + len(history),
                return_sequence=True,
            )
        else:
            # Diagnostic upper bound only: this sequence uses realized future
            # states and must never be presented as deployable input.
            baseline_env, baseline_obs, baseline_reference = replay_prefix(
                domain, q0a, target, locked, args.approach_steps, history
            )
            carrier_sequence = []
            for offset in range(args.horizon):
                step = len(history) + offset
                nominal = joint_reference_action(
                    baseline_obs["state"][:10], baseline_reference, locked_joints=locked
                )
                action = carrier_proposal(
                    carrier_wrapped, baseline_obs, nominal, target, mask, angle,
                    baseline_env, locked, candidates=args.planner_candidates,
                    horizon=args.horizon, iterations=args.iterations,
                    seed=args.seed * 10_000 + target_index * 100 + step,
                    guard_threshold=args.guard_threshold,
                )
                carrier_sequence.append(action.copy())
                baseline_obs = baseline_env.step(action)["observation"]
            carrier_sequence = np.asarray(carrier_sequence)

        rng = np.random.default_rng(args.seed * 100_000 + target_index)
        candidates = np.clip(
            carrier_sequence[None] + args.noise_std * rng.standard_normal(
                (args.candidates, args.horizon, 5)
            ), -1.0, 1.0,
        )
        candidates[0] = carrier_sequence
        if locked:
            candidates[:, :, list(locked)] = 0.0
        carrier_pred = predicted_costs(
            carrier_wrapped, state, candidates, target[:2], mask, angle
        )
        selective_pred = predicted_costs(
            selective, state, candidates, target[:2], mask, angle
        )
        terminal = []
        for index, sequence in enumerate(candidates):
            result = terminal_rollout(
                domain, q0a, target, locked, args.approach_steps, history, sequence,
                carrier_wrapped, mask, angle, target_index=target_index,
                total_push_steps=args.total_push_steps,
                planner_candidates=args.planner_candidates,
                planner_horizon=args.horizon, planner_iterations=args.iterations,
                guard_threshold=args.guard_threshold, seed=args.seed,
            )
            terminal.append(result["final_cost_m"])
            delta = sequence - carrier_sequence
            rows.append({
                "target": f"{item.target_id}@terminal", "base_target": item.target_id,
                "target_split": args.target_split, "candidate_index": index,
                "state": state.tolist(),
                "carrier_sequence": carrier_sequence.reshape(-1).tolist(),
                "candidate_sequence": sequence.reshape(-1).tolist(),
                "candidate_delta": delta.reshape(-1).tolist(),
                "true_cost_m": result["final_cost_m"],
                "short_carrier_predicted_cost_m": float(carrier_pred[index]),
                "short_selective_predicted_cost_m": float(selective_pred[index]),
                "carrier_predicted_cost_m": float(carrier_pred[index]),
                "selective_predicted_cost_m": float(selective_pred[index]),
                "predicted_cost_delta_m": float(selective_pred[index] - carrier_pred[index]),
                "action_deviation_rms": float(np.sqrt(np.mean(delta ** 2))),
                "first_action_deviation_l2": float(np.linalg.norm(delta[0])),
                "action_effort_rms": float(np.sqrt(np.mean(sequence ** 2))),
                "tool_block_distance_m": float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos())),
                "goal_distance_m": float(np.linalg.norm(env.block_pos() - target[:2])),
                "block_speed_mps": float(np.linalg.norm(state[12:14])),
                "nominal_action_l2": float(np.linalg.norm(joint_reference_action(
                    state[:10], push_reference, locked_joints=locked
                ))),
                **result,
            })
            print(f"[{item.target_id}] candidate {index + 1}/{args.candidates}: {result['final_cost_m']:.6f}", flush=True)
        terminal = np.asarray(terminal)
        summaries.append({
            "target": item.target_id, "contact_step": len(history),
            "candidate_zero_cost_m": float(terminal[0]),
            "oracle_cost_m": float(terminal.min()),
            "oracle_index": int(terminal.argmin()),
            "oracle_improvement_pct": float(
                100.0 * (terminal[0] - terminal.min()) / max(terminal[0], 1e-8)
            ),
            "candidate_zero_reproduces_carrier":
                args.base_sequence_mode == "receding_oracle_audit",
            "base_sequence_mode": args.base_sequence_mode,
            "deployable_base_sequence": args.base_sequence_mode == "one_shot_cem",
        })
    payload = {
        "version": "terminal_action_ranker_branches_v2", "development_only": True,
        "seed": args.seed, "domain": args.domain, "target_split": args.target_split,
        "protocol": {"candidates": args.candidates,
                     "planner_candidates": args.planner_candidates,
                     "horizon": args.horizon, "total_push_steps": args.total_push_steps,
                     "noise_std": args.noise_std,
                     "base_sequence_mode": args.base_sequence_mode,
                     "uses_realized_future_state_for_base":
                         args.base_sequence_mode == "receding_oracle_audit"},
        "summaries": summaries, "candidate_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
