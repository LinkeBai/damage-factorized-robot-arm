"""Diagnose whether lower SI-IPWM rollout RMSE yields better MPC action ranking.

Development-only diagnostic.  It evaluates carrier and selective IPWM on the
same candidate action sequences and compares each predicted terminal block cost
with the realized MuJoCo cost.  It does not tune or report a controller win.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import FewShotProjectedModel
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.evaluate_ipwm_support_validation_gate import build_strict, make_adapter
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import PUSH_WAYPOINT_OFFSET, PUSH_XML, collect_push_domains


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, without a scipy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a, b = rankdata(first), rankdata(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_models(cfg: dict, seed: int, model_path: Path, device: torch.device):
    full = build_strict(cfg, device)
    full.load_state_dict(torch.load(model_path, map_location=device))
    carrier = copy.deepcopy(full)
    with torch.no_grad():
        for name in ("geometric_object_head", "intervention_object_head"):
            if hasattr(carrier, name):
                for parameter in getattr(carrier, name).parameters():
                    parameter.zero_()
    adapter_cfg = yaml.safe_load(Path(cfg["matched_adapter_config"]).read_text(encoding="utf-8"))
    adapter_state = torch.load(
        Path(str(cfg["matched_adapter_run_template"]).format(seed=seed)) / "bt_adapter.pt",
        map_location=device,
    )
    full_adapter, carrier_adapter = make_adapter(adapter_cfg, device), make_adapter(adapter_cfg, device)
    full_adapter.load_state_dict(adapter_state)
    carrier_adapter.load_state_dict(adapter_state)
    full_wrapped = FewShotProjectedModel(
        full, full_adapter, base_uses_topology=True
    ).to(device).eval()
    carrier_wrapped = FewShotProjectedModel(
        carrier, carrier_adapter, base_uses_topology=True
    ).to(device).eval()
    selective = SelectiveInterventionRollout(
        full_wrapped, carrier_wrapped
    ).to(device).eval()
    encoder = UncertainPhysicalContextEncoder(
        hidden_dim=int(cfg.get("context_encoder_hidden_dim", 96))
    ).to(device)
    encoder.load_state_dict(torch.load(
        Path(str(cfg["context_encoder_run_template"]).format(seed=seed)) / "context_encoder.pt",
        map_location=device,
    ))
    encoder.eval()
    return full, carrier, carrier_wrapped, selective, encoder


@torch.no_grad()
def predicted_costs(model, state, actions, target_xy, mask, angle):
    device = next(model.parameters()).device
    batch = len(actions)
    simulated = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1)
    mask_batch, angle_batch = mask.expand(batch, -1), angle.expand(batch, -1)
    hidden = None
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    for depth in range(actions.shape[1]):
        simulated, hidden = model.step(simulated, action_tensor[:, depth], mask_batch, angle_batch, hidden)
    target = torch.as_tensor(target_xy, dtype=torch.float32, device=device)
    return torch.linalg.vector_norm(simulated[:, 10:12] - target, dim=-1).cpu().numpy()


def replay_to_decision(domain, q0a, target, locked, approach_steps, nominal_push_warmup):
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
    for _ in range(approach_steps):
        action = joint_reference_action(observation["state"][:10], approach_reference, locked_joints=locked)
        observation = env.step(action)["observation"]
    push_reference, _ = solve_reach_reference(
        target + PUSH_WAYPOINT_OFFSET, env.joint_ranges, locked_joints=lock_map,
    )
    for _ in range(nominal_push_warmup):
        nominal = joint_reference_action(observation["state"][:10], push_reference, locked_joints=locked)
        observation = env.step(nominal)["observation"]
    nominal = joint_reference_action(observation["state"][:10], push_reference, locked_joints=locked)
    return env, observation["state"].copy(), nominal


def realized_costs(domain, q0a, target, locked, approach_steps, nominal_push_warmup, candidates):
    costs = []
    for sequence in candidates:
        env, _, _ = replay_to_decision(
            domain, q0a, target, locked, approach_steps, nominal_push_warmup
        )
        for action in sequence:
            env.step(action)
        costs.append(float(np.linalg.norm(env.block_pos() - target[:2])))
    return np.asarray(costs)


def first_nominal_contact_step(domain, q0a, target, locked, approach_steps, max_steps=90):
    env, state, _ = replay_to_decision(domain, q0a, target, locked, approach_steps, 0)
    lock_map = {i: domain.damage.lock_angle_of(i) for i in locked}
    push_reference, _ = solve_reach_reference(
        target + PUSH_WAYPOINT_OFFSET, env.joint_ranges, locked_joints=lock_map,
    )
    for step in range(max_steps):
        nominal = joint_reference_action(state[:10], push_reference, locked_joints=locked)
        result = env.step(nominal)
        state = result["observation"]["state"]
        if env.last_has_contact("tool_geom", "block_geom") or env.last_has_contact("pusher_geom", "block_geom"):
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--domain", default="D3__high_damping")
    parser.add_argument(
        "--target-split", choices=("calibration", "validation", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--target-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--approach-steps", type=int, default=60)
    parser.add_argument("--nominal-push-warmup", type=int, default=0)
    parser.add_argument("--auto-before-first-contact", action="store_true")
    parser.add_argument(
        "--contact-offsets", nargs="+", type=int,
        help="Decision warmups relative to the first nominal contact step.",
    )
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.35)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    domains = {d.domain_id: d for d in (*protocol.train, *protocol.validation, *protocol.test)}
    domain = domains[args.domain]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, carrier, carrier_wrapped, selective, encoder = build_models(cfg, args.seed, args.model, device)

    calibration_seed = args.seed * 100_000 + list(domains).index(args.domain) * 1000 + 100
    calibration_key = json.dumps({
        "kind": "push_context_calibration", "seed": calibration_seed,
        "domain": domain.domain_id, "budget": 25, "q0a": q0a,
    }, sort_keys=True)
    calibration = cached_collect(
        args.cache_dir, calibration_key,
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
    rows, candidate_rows = [], []
    target_pool = getattr(targets, args.target_split)
    for target_index in args.target_indices:
        target_item = target_pool[target_index]
        target = target_item.as_array()
        first_contact = first_nominal_contact_step(
            domain, q0a, target, locked, args.approach_steps
        )
        warmups = [args.nominal_push_warmup]
        if args.auto_before_first_contact and first_contact is not None:
            warmups = [max(0, first_contact - args.horizon + 1)]
        if args.contact_offsets is not None:
            if first_contact is None:
                raise RuntimeError(f"no nominal contact found for {target_item.target_id}")
            warmups = [max(0, first_contact + offset) for offset in args.contact_offsets]
        for phase_index, warmup in enumerate(dict.fromkeys(warmups)):
            env, state, nominal = replay_to_decision(
                domain, q0a, target, locked, args.approach_steps, warmup
            )
            tool_block_distance = float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos()))
            goal_distance = float(np.linalg.norm(env.block_pos() - target[:2]))
            block_speed = float(np.linalg.norm(state[12:14]))
            current_contact = bool(
                env.has_contact("tool_geom", "block_geom")
                or env.has_contact("pusher_geom", "block_geom")
            )
            rng = np.random.default_rng(args.seed * 100000 + target_index * 100 + phase_index)
            actions = np.clip(
                nominal.reshape(1, 1, -1)
                + args.noise_std * rng.standard_normal((args.candidates, args.horizon, 5)),
                -1.0, 1.0,
            )
            actions[0] = nominal.reshape(1, -1)
            if locked:
                actions[:, :, list(locked)] = 0.0
            true = realized_costs(
                domain, q0a, target, locked, args.approach_steps, warmup, actions,
            )
            carrier_pred = predicted_costs(carrier_wrapped, state, actions, target[:2], mask, angle)
            selective_pred = predicted_costs(selective, state, actions, target[:2], mask, angle)
            decision_id = f"{target_item.target_id}@w{warmup}"
            for candidate_index in range(args.candidates):
                deviation = actions[candidate_index] - nominal.reshape(1, -1)
                candidate_rows.append({
                    "target": decision_id, "base_target": target_item.target_id,
                    "target_split": args.target_split, "candidate_index": candidate_index,
                    "decision_warmup_steps": warmup,
                    "tool_block_distance_m": tool_block_distance,
                    "goal_distance_m": goal_distance, "block_speed_mps": block_speed,
                    "current_contact": current_contact,
                    "nominal_action_l2": float(np.linalg.norm(nominal)),
                    "true_cost_m": float(true[candidate_index]),
                    "carrier_predicted_cost_m": float(carrier_pred[candidate_index]),
                    "selective_predicted_cost_m": float(selective_pred[candidate_index]),
                    "predicted_cost_delta_m": float(selective_pred[candidate_index] - carrier_pred[candidate_index]),
                    "action_deviation_rms": float(np.sqrt(np.mean(deviation ** 2))),
                    "first_action_deviation_l2": float(np.linalg.norm(deviation[0])),
                    "action_effort_rms": float(np.sqrt(np.mean(actions[candidate_index] ** 2))),
                })
            oracle = int(np.argmin(true))
            for name, pred in (("carrier", carrier_pred), ("selective_ipwm", selective_pred)):
                chosen = int(np.argmin(pred))
                top_quartile = set(np.argsort(pred)[: max(1, args.candidates // 4)])
                rows.append({
                    "target": target_item.target_id, "decision_id": decision_id, "model": name,
                    "first_nominal_contact_step": first_contact,
                    "decision_warmup_steps": warmup,
                    "decision_tool_block_distance_m": tool_block_distance,
                    "goal_distance_m": goal_distance, "block_speed_mps": block_speed,
                    "current_contact": current_contact,
                    "spearman": spearman(pred, true),
                    "pearson": float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) and np.std(true) else 0.0,
                    "chosen_index": chosen, "oracle_index": oracle,
                    "chosen_realized_cost_m": float(true[chosen]),
                    "oracle_realized_cost_m": float(true[oracle]),
                    "regret_m": float(true[chosen] - true[oracle]),
                    "oracle_in_predicted_top_quartile": oracle in top_quartile,
                    "predicted_cost_std_m": float(np.std(pred)),
                    "realized_cost_std_m": float(np.std(true)),
                })
                print(rows[-1], flush=True)
    payload = {
        "version": "ipwm_action_ranking_diagnostic_v1", "development_only": True,
        "seed": args.seed, "domain": args.domain, "target_split": args.target_split,
        "context_norm": float(context.norm().cpu()),
        "protocol": {"targets": args.target_indices, "approach_steps": args.approach_steps,
                     "nominal_push_warmup": args.nominal_push_warmup,
                     "auto_before_first_contact": args.auto_before_first_contact,
                     "contact_offsets": args.contact_offsets,
                     "candidates": args.candidates, "horizon": args.horizon, "noise_std": args.noise_std},
        "rows": rows, "candidate_rows": candidate_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
