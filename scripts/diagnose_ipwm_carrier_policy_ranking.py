"""Collect contact-local counterfactual branches on the carrier-MPC state distribution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.diagnose_ipwm_action_ranking import build_models, predicted_costs, spearman
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_ipwm_closed_loop_audit import plan_push
from scripts.run_push_benchmark import PUSH_WAYPOINT_OFFSET, PUSH_XML, collect_push_domains


def initialize_env(domain, q0a, target, locked, approach_steps):
    env = MujocoArmEnv(
        xml_path=PUSH_XML, residual_physics=domain.residual,
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
    )
    observation = env.reset(target=target, damage_config=domain.damage)
    initial = env.block_pos().copy()
    lock_map = {i: domain.damage.lock_angle_of(i) for i in locked}
    approach_reference, _ = solve_reach_reference(
        np.array([initial[0] - 0.03, initial[1], 0.025]),
        env.joint_ranges, locked_joints=lock_map,
    )
    push_reference, _ = solve_reach_reference(
        target + PUSH_WAYPOINT_OFFSET, env.joint_ranges, locked_joints=lock_map,
    )
    for _ in range(approach_steps):
        action = joint_reference_action(observation["state"][:10], approach_reference, locked_joints=locked)
        observation = env.step(action)["observation"]
    return env, observation, push_reference


def has_contact(env):
    return bool(
        env.has_contact("tool_geom", "block_geom")
        or env.has_contact("pusher_geom", "block_geom")
        or env.last_has_contact("tool_geom", "block_geom")
        or env.last_has_contact("pusher_geom", "block_geom")
    )


def replay_cost(domain, q0a, target, locked, approach_steps, history, sequence):
    env, observation, _ = initialize_env(domain, q0a, target, locked, approach_steps)
    for action in history:
        observation = env.step(action)["observation"]
    for action in sequence:
        observation = env.step(action)["observation"]
    return float(np.linalg.norm(env.block_pos() - target[:2]))


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
    parser.add_argument("--max-push-steps", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--guard-threshold", type=float, default=0.85)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    domains = {d.domain_id: d for d in (*protocol.train, *protocol.validation, *protocol.test)}
    domain = domains[args.domain]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, carrier, carrier_wrapped, selective, encoder = build_models(cfg, args.seed, args.model, device)

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
        context, _ = encoder(calibration.states[None].to(device), calibration.actions[None].to(device), mask, return_uncertainty=True)
    context = context[0] * float(cfg.get("context_posterior_scale", 1.0))
    full.set_intervention_context(context)
    carrier.set_intervention_context(context)
    selective.set_residual_context(context)

    locked = tuple(domain.damage.locked)
    pool = getattr(targets, args.target_split)
    summaries, candidate_rows = [], []
    for target_index in args.target_indices:
        item, target = pool[target_index], pool[target_index].as_array()
        env, observation, push_reference = initialize_env(
            domain, q0a, target, locked, args.approach_steps
        )
        history = []
        for step in range(args.max_push_steps):
            if has_contact(env):
                break
            nominal = joint_reference_action(observation["state"][:10], push_reference, locked_joints=locked)
            sequence = plan_push(
                carrier_wrapped, observation["state"], nominal, target[:2], mask, angle,
                env.joint_ranges, locked, candidates=args.candidates, horizon=args.horizon,
                iterations=args.iterations, seed=args.seed * 10_000 + target_index * 100 + step,
                return_sequence=True,
            )
            action = sequence[0]
            if np.linalg.norm(action - nominal) > args.guard_threshold:
                action = nominal
            history.append(action.copy())
            observation = env.step(action)["observation"]
        if not has_contact(env):
            raise RuntimeError(f"carrier policy did not contact for {item.target_id}")

        state = observation["state"].copy()
        nominal = joint_reference_action(state[:10], push_reference, locked_joints=locked)
        carrier_sequence = plan_push(
            carrier_wrapped, state, nominal, target[:2], mask, angle,
            env.joint_ranges, locked, candidates=args.candidates, horizon=args.horizon,
            iterations=args.iterations, seed=args.seed * 10_000 + target_index * 100 + len(history),
            return_sequence=True,
        )
        rng = np.random.default_rng(args.seed * 100_000 + target_index)
        actions = np.clip(
            carrier_sequence[None] + args.noise_std * rng.standard_normal(
                (args.candidates, args.horizon, 5)
            ), -1.0, 1.0,
        )
        actions[0] = carrier_sequence
        if locked:
            actions[:, :, list(locked)] = 0.0
        true = np.asarray([
            replay_cost(domain, q0a, target, locked, args.approach_steps, history, sequence)
            for sequence in actions
        ])
        carrier_pred = predicted_costs(carrier_wrapped, state, actions, target[:2], mask, angle)
        selective_pred = predicted_costs(selective, state, actions, target[:2], mask, angle)
        decision_id = f"{item.target_id}@carrier_contact_{len(history)}"
        for index in range(args.candidates):
            deviation = actions[index] - carrier_sequence
            candidate_rows.append({
                "target": decision_id, "base_target": item.target_id,
                "target_split": args.target_split, "candidate_index": index,
                "carrier_contact_step": len(history),
                "tool_block_distance_m": float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos())),
                "goal_distance_m": float(np.linalg.norm(env.block_pos() - target[:2])),
                "block_speed_mps": float(np.linalg.norm(state[12:14])),
                "current_contact": True,
                "nominal_action_l2": float(np.linalg.norm(nominal)),
                "state": state.tolist(),
                "carrier_sequence": carrier_sequence.reshape(-1).tolist(),
                "candidate_sequence": actions[index].reshape(-1).tolist(),
                "candidate_delta": deviation.reshape(-1).tolist(),
                "true_cost_m": float(true[index]),
                "carrier_predicted_cost_m": float(carrier_pred[index]),
                "selective_predicted_cost_m": float(selective_pred[index]),
                "predicted_cost_delta_m": float(selective_pred[index] - carrier_pred[index]),
                "action_deviation_rms": float(np.sqrt(np.mean(deviation ** 2))),
                "first_action_deviation_l2": float(np.linalg.norm(deviation[0])),
                "action_effort_rms": float(np.sqrt(np.mean(actions[index] ** 2))),
            })
        for name, pred in (("carrier", carrier_pred), ("selective_ipwm", selective_pred)):
            chosen, oracle = int(np.argmin(pred)), int(np.argmin(true))
            row = {
                "target": decision_id, "model": name, "contact_step": len(history),
                "spearman": spearman(pred, true), "chosen_index": chosen,
                "oracle_index": oracle, "chosen_realized_cost_m": float(true[chosen]),
                "carrier_sequence_realized_cost_m": float(true[0]),
                "oracle_realized_cost_m": float(true[oracle]),
                "regret_m": float(true[chosen] - true[oracle]),
            }
            summaries.append(row)
            print(row, flush=True)
    payload = {
        "version": "ipwm_carrier_policy_ranking_v1", "development_only": True,
        "seed": args.seed, "domain": args.domain, "target_split": args.target_split,
        "protocol": {"candidates": args.candidates, "horizon": args.horizon,
                     "iterations": args.iterations, "noise_std": args.noise_std},
        "rows": summaries, "candidate_rows": candidate_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
