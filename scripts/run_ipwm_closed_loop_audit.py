"""Frozen guarded-CEM Push audit for carrier and selective IPWM dynamics."""
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
from robotarm.models.contact_action_ranker import ContactActionRanker
from robotarm.models.projected_residual_innovation import FewShotProjectedModel
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.controllers import (
    directional_push_waypoints,
    joint_reference_action,
    solve_reach_reference,
)
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.evaluate_ipwm_support_validation_gate import build_strict, make_adapter
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import (
    PUSH_XML,
    collect_push_domains,
)


@torch.no_grad()
def plan_push(
    model,
    state: np.ndarray,
    nominal_action: np.ndarray,
    target_xy: np.ndarray,
    mask: torch.Tensor,
    angle: torch.Tensor,
    joint_ranges: np.ndarray,
    locked: tuple[int, ...],
    *,
    candidates: int,
    horizon: int,
    iterations: int,
    seed: int,
    return_sequence: bool = False,
) -> np.ndarray:
    device = next(model.parameters()).device
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    nominal = torch.as_tensor(nominal_action, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_xy, dtype=torch.float32, device=device).reshape(1, 2)
    ranges = torch.as_tensor(joint_ranges, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = nominal.reshape(1, -1).expand(horizon, -1).clone()
    std = torch.full_like(mean, 0.35)
    mask_batch = mask.expand(candidates, -1)
    angle_batch = angle.expand(candidates, -1)
    for _ in range(iterations):
        actions = torch.clamp(
            mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                candidates, horizon, 5, generator=generator, device=device
            ), -1.0, 1.0,
        )
        if locked:
            actions[:, :, list(locked)] = 0.0
        simulated = state_t.expand(candidates, -1)
        hidden = None
        for depth in range(horizon):
            simulated, hidden = model.step(
                simulated, actions[:, depth], mask_batch, angle_batch, hidden
            )
        block_cost = torch.linalg.vector_norm(simulated[:, 10:12] - target, dim=-1)
        qpos = simulated[:, :5]
        violation = torch.relu(ranges[:, 0] - qpos).pow(2)
        violation += torch.relu(qpos - ranges[:, 1]).pow(2)
        deviation = (actions - nominal.reshape(1, 1, -1)).pow(2).mean(dim=(1, 2))
        effort = actions.pow(2).mean(dim=(1, 2))
        cost = block_cost + 5.0 * violation.mean(dim=-1) + 0.03 * deviation + 0.005 * effort
        elite = actions[torch.topk(cost, max(2, candidates // 4), largest=False).indices]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp_min(0.05)
    sequence = mean.clamp(-1.0, 1.0).cpu().numpy()
    if locked:
        sequence[:, list(locked)] = 0.0
    return sequence if return_sequence else sequence[0]


@torch.no_grad()
def plan_push_carrier_screened(
    carrier,
    intervention,
    state: np.ndarray,
    nominal_action: np.ndarray,
    target_xy: np.ndarray,
    mask: torch.Tensor,
    angle: torch.Tensor,
    joint_ranges: np.ndarray,
    locked: tuple[int, ...],
    *,
    candidates: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Let intervention rerank only the carrier's top-quartile candidates."""
    device = next(carrier.parameters()).device
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    nominal = torch.as_tensor(nominal_action, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_xy, dtype=torch.float32, device=device).reshape(1, 2)
    ranges = torch.as_tensor(joint_ranges, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.clamp(
        nominal.reshape(1, 1, -1).expand(candidates, horizon, -1)
        + 0.35 * torch.randn(candidates, horizon, 5, generator=generator, device=device),
        -1.0, 1.0,
    )
    actions[0] = nominal.reshape(1, -1).expand(horizon, -1)
    if locked:
        actions[:, :, list(locked)] = 0.0
    mask_batch, angle_batch = mask.expand(candidates, -1), angle.expand(candidates, -1)

    def costs(model):
        simulated, hidden = state_t.expand(candidates, -1), None
        for depth in range(horizon):
            simulated, hidden = model.step(
                simulated, actions[:, depth], mask_batch, angle_batch, hidden
            )
        block = torch.linalg.vector_norm(simulated[:, 10:12] - target, dim=-1)
        qpos = simulated[:, :5]
        violation = torch.relu(ranges[:, 0] - qpos).pow(2)
        violation += torch.relu(qpos - ranges[:, 1]).pow(2)
        deviation = (actions - nominal.reshape(1, 1, -1)).pow(2).mean(dim=(1, 2))
        effort = actions.pow(2).mean(dim=(1, 2))
        return block + 5.0 * violation.mean(dim=-1) + 0.03 * deviation + 0.005 * effort

    carrier_cost = costs(carrier)
    safe = torch.topk(carrier_cost, max(2, candidates // 4), largest=False).indices
    if intervention is None:
        chosen = safe[0]
    else:
        intervention_cost = costs(intervention)
        chosen = safe[torch.argmin(intervention_cost[safe])]
    action = actions[chosen, 0].clamp(-1.0, 1.0).cpu().numpy()
    if locked:
        action[list(locked)] = 0.0
    return action


@torch.no_grad()
def plan_push_calibrated(
    carrier, intervention, ranker, state, nominal_action, target_xy, mask, angle,
    locked, *, candidates, horizon, seed, noise_std, return_sequence=False,
    base_sequence=None, score_mode="calibrated",
) -> np.ndarray:
    """Rank local candidates with a calibration model fitted on branch rollouts."""
    device = next(carrier.parameters()).device
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    nominal = torch.as_tensor(nominal_action, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_xy, dtype=torch.float32, device=device).reshape(1, 2)
    generator = torch.Generator(device=device).manual_seed(seed)
    base = (
        nominal.reshape(1, -1).expand(horizon, -1)
        if base_sequence is None
        else torch.as_tensor(base_sequence, dtype=torch.float32, device=device)
    )
    actions = torch.clamp(
        base.unsqueeze(0).expand(candidates, -1, -1)
        + noise_std * torch.randn(candidates, horizon, 5, generator=generator, device=device),
        -1.0, 1.0,
    )
    actions[0] = base
    if locked:
        actions[:, :, list(locked)] = 0.0
    mask_batch, angle_batch = mask.expand(candidates, -1), angle.expand(candidates, -1)

    def terminal_cost(model):
        simulated, hidden = state_t.expand(candidates, -1), None
        for depth in range(horizon):
            simulated, hidden = model.step(
                simulated, actions[:, depth], mask_batch, angle_batch, hidden
            )
        return torch.linalg.vector_norm(simulated[:, 10:12] - target, dim=-1)

    carrier_cost = terminal_cost(carrier)
    selective_cost = terminal_cost(intervention)
    deviation = actions - base.unsqueeze(0)
    features = torch.stack((
        carrier_cost,
        selective_cost,
        selective_cost - carrier_cost,
        torch.sqrt(torch.mean(deviation.pow(2), dim=(1, 2))),
        torch.linalg.vector_norm(deviation[:, 0], dim=-1),
        torch.sqrt(torch.mean(actions.pow(2), dim=(1, 2))),
    ), dim=1)
    if score_mode == "selective":
        score = selective_cost
    else:
        coefficient = torch.as_tensor(ranker["coefficient"], dtype=features.dtype, device=device)
        scale = torch.as_tensor(ranker["feature_scale"], dtype=features.dtype, device=device)
        centered = features - features.mean(dim=0, keepdim=True)
        score = (centered / scale) @ coefficient
    chosen = int(torch.argmin(score).item())
    selected = actions[chosen].clamp(-1.0, 1.0).cpu().numpy()
    if locked:
        selected[:, list(locked)] = 0.0
    return selected if return_sequence else selected[0]


@torch.no_grad()
def plan_push_neural_ranker(
    carrier, intervention, neural_ranker, state, nominal_action, target_xy,
    mask, angle, locked, *, candidates, horizon, seed, noise_std,
    base_sequence, tool_block_distance,
) -> np.ndarray:
    device = next(carrier.parameters()).device
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    nominal = torch.as_tensor(nominal_action, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_xy, dtype=torch.float32, device=device)
    base = torch.as_tensor(base_sequence, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.clamp(
        base.unsqueeze(0).expand(candidates, -1, -1)
        + noise_std * torch.randn(candidates, horizon, 5, generator=generator, device=device),
        -1.0, 1.0,
    )
    actions[0] = base
    if locked:
        actions[:, :, list(locked)] = 0.0
    mask_batch, angle_batch = mask.expand(candidates, -1), angle.expand(candidates, -1)

    def terminal_cost(model):
        simulated = state_t.reshape(1, -1).expand(candidates, -1)
        hidden = None
        for depth in range(horizon):
            simulated, hidden = model.step(
                simulated, actions[:, depth], mask_batch, angle_batch, hidden
            )
        return torch.linalg.vector_norm(simulated[:, 10:12] - target, dim=-1)

    carrier_cost, selective_cost = terminal_cost(carrier), terminal_cost(intervention)
    delta = actions - base.unsqueeze(0)
    scalars = torch.stack((
        carrier_cost, selective_cost, selective_cost - carrier_cost,
        torch.sqrt(torch.mean(delta.pow(2), dim=(1, 2))),
        torch.linalg.vector_norm(delta[:, 0], dim=-1),
        torch.sqrt(torch.mean(actions.pow(2), dim=(1, 2))),
        torch.full((candidates,), float(tool_block_distance), device=device),
        torch.full((candidates,), float(torch.linalg.vector_norm(state_t[10:12] - target)), device=device),
        torch.full((candidates,), float(torch.linalg.vector_norm(state_t[12:14])), device=device),
        torch.full((candidates,), float(torch.linalg.vector_norm(nominal)), device=device),
    ), dim=1)
    features = torch.cat((
        state_t.reshape(1, -1).expand(candidates, -1),
        base.reshape(1, -1).expand(candidates, -1),
        delta.flatten(1), scalars,
    ), dim=1)
    normalized = (features - neural_ranker["mean"]) / neural_ranker["scale"]
    score = neural_ranker["model"](normalized)
    selected = actions[int(torch.argmin(score).item())].cpu().numpy()
    if locked:
        selected[:, list(locked)] = 0.0
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--domains", nargs="+", default=["D3__mixed_unseen"])
    parser.add_argument("--target-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--approach-steps", type=int, default=30)
    parser.add_argument("--push-steps", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--guard-threshold", type=float, default=0.85)
    parser.add_argument("--physical-threshold", type=float, default=1.2)
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--neural-ranker", type=Path)
    parser.add_argument("--ranker-noise-std", type=float, default=0.10)
    parser.add_argument("--ranker-contact-distance", type=float, default=0.06)
    parser.add_argument("--methods", nargs="+", help="Optional method-name subset.")
    parser.add_argument("--context-budget", type=int, default=25)
    args = parser.parse_args()
    ranker = None if args.ranker is None else json.loads(args.ranker.read_text(encoding="utf-8"))
    neural_ranker = None
    if args.neural_ranker is not None:
        checkpoint = torch.load(args.neural_ranker, map_location="cpu")
        neural_model = ContactActionRanker(
            int(checkpoint["input_dim"]), tuple(checkpoint["hidden_dims"])
        )
        neural_model.load_state_dict(checkpoint["state_dict"])
        neural_ranker = {"model": neural_model, "checkpoint": checkpoint}

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if neural_ranker is not None:
        neural_ranker["model"] = neural_ranker["model"].to(device).eval()
        neural_ranker["mean"] = torch.as_tensor(
            neural_ranker["checkpoint"]["normalization_mean"], dtype=torch.float32, device=device
        )
        neural_ranker["scale"] = torch.as_tensor(
            neural_ranker["checkpoint"]["normalization_scale"], dtype=torch.float32, device=device
        )

    full = build_strict(cfg, device)
    full.load_state_dict(torch.load(args.model, map_location=device))
    carrier = copy.deepcopy(full)
    with torch.no_grad():
        for name in ("geometric_object_head", "intervention_object_head"):
            if hasattr(carrier, name):
                for parameter in getattr(carrier, name).parameters():
                    parameter.zero_()
    adapter_cfg = yaml.safe_load(Path(cfg["matched_adapter_config"]).read_text(encoding="utf-8"))
    adapter_state = torch.load(
        Path(str(cfg["matched_adapter_run_template"]).format(seed=args.seed)) / "bt_adapter.pt",
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
        Path(str(cfg["context_encoder_run_template"]).format(seed=args.seed)) / "context_encoder.pt",
        map_location=device,
    ))
    encoder.eval()

    requested = set(args.domains)
    all_domains = tuple(dict.fromkeys((*protocol.train, *protocol.validation, *protocol.test)))
    rows = []
    routing = []
    for domain_index, domain in enumerate(all_domains):
        if domain.domain_id not in requested:
            continue
        calibration_seed = args.seed * 100_000 + domain_index * 1000 + 100
        key = json.dumps({
            "kind": "push_context_calibration", "seed": calibration_seed,
            "domain": domain.domain_id, "budget": args.context_budget, "q0a": q0a,
        }, sort_keys=True)
        calibration = cached_collect(
            args.cache_dir, key,
            lambda domain=domain: collect_push_domains(
                (domain,), trajectories_per_domain=1, steps=args.context_budget,
                seed=calibration_seed,
                targets=tuple(x.as_array() for x in targets.calibration),
                excitation="active",
                block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
                goal_exploration_std=float(q0a["goal_exploration_std"]),
            ),
        )[0]
        mask, angle = _damage_tensors([domain.damage], device)
        with torch.no_grad():
            context, _ = encoder(
                calibration.states[None].to(device),
                calibration.actions[None].to(device), mask,
                return_uncertainty=True,
            )
        context = context[0] * float(cfg.get("context_posterior_scale", 1.0))
        full.set_intervention_context(context)
        carrier.set_intervention_context(context)
        selective.set_residual_context(context)
        active = float(context.norm().cpu()) >= args.physical_threshold
        routing.append({"domain": domain.domain_id, "context_norm": float(context.norm().cpu()), "active": active})
        locked = tuple(domain.damage.locked)
        for target_index in args.target_indices:
            target_item = targets.evaluation[target_index]
            target = target_item.as_array()
            methods = [
                ("nominal_ik", None),
                ("carrier_guarded_mpc", carrier_wrapped),
                ("selective_ipwm_guarded_mpc", selective if active else carrier_wrapped),
                ("carrier_screen_mpc", "carrier_screen"),
                ("selective_ipwm_rerank_mpc", "selective_rerank"),
            ]
            if ranker is not None:
                methods.append(("calibrated_ipwm_contact_mpc", "calibrated_ranker"))
                methods.append(("calibrated_ipwm_one_shot_sequence", "calibrated_sequence"))
                methods.append(("calibrated_ipwm_contact_event_sequence", "contact_event_sequence"))
                methods.append(("carrier_ipwm_residual_sequence", "carrier_residual_sequence"))
                methods.append(("carrier_ipwm_selective_sequence", "carrier_selective_sequence"))
            if neural_ranker is not None:
                methods.append(("carrier_neural_residual_sequence", "carrier_neural_sequence"))
            if args.methods:
                requested_methods = set(args.methods)
                methods = [item for item in methods if item[0] in requested_methods]
                unknown = requested_methods - {item[0] for item in methods}
                if unknown:
                    raise ValueError(f"unknown or unavailable methods: {sorted(unknown)}")
            for method, model in methods:
                env = MujocoArmEnv(
                    xml_path=PUSH_XML,
                    residual_physics=domain.residual,
                    block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
                )
                observation = env.reset(target=target, damage_config=domain.damage)
                initial = env.block_pos().copy()
                lock_map = {i: domain.damage.lock_angle_of(i) for i in locked}
                approach_reference, _ = solve_reach_reference(
                    directional_push_waypoints(initial, target[:2])[0],
                    env.joint_ranges, locked_joints=lock_map,
                )
                push_reference, _ = solve_reach_reference(
                    directional_push_waypoints(initial, target[:2])[1],
                    env.joint_ranges,
                    locked_joints=lock_map,
                )
                for _ in range(args.approach_steps):
                    action = joint_reference_action(
                        observation["state"][:10], approach_reference, locked_joints=locked
                    )
                    observation = env.step(action)["observation"]
                contacts = fallbacks = 0
                deviations = []
                planned_sequence = []
                ranker_triggered = False
                for step in range(args.push_steps):
                    nominal = joint_reference_action(
                        observation["state"][:10], push_reference, locked_joints=locked
                    )
                    action = nominal
                    if model is not None:
                        plan_seed = args.seed * 10_000 + target_index * 100 + step
                        if model in ("carrier_screen", "selective_rerank"):
                            proposed = plan_push_carrier_screened(
                                carrier_wrapped,
                                selective if model == "selective_rerank" and active else None,
                                observation["state"], nominal, target[:2], mask, angle,
                                env.joint_ranges, locked, candidates=args.candidates,
                                horizon=args.horizon, seed=plan_seed,
                            )
                        elif model == "calibrated_ranker":
                            tool_block_distance = float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos()))
                            if active and tool_block_distance <= args.ranker_contact_distance:
                                proposed = plan_push_calibrated(
                                    carrier_wrapped, selective, ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                )
                            else:
                                proposed = nominal
                        elif model == "calibrated_sequence":
                            tool_block_distance = float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos()))
                            if planned_sequence:
                                proposed = planned_sequence.pop(0)
                            elif (not ranker_triggered and active
                                  and tool_block_distance <= args.ranker_contact_distance):
                                sequence = plan_push_calibrated(
                                    carrier_wrapped, selective, ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                    return_sequence=True,
                                )
                                proposed = sequence[0]
                                planned_sequence = [item for item in sequence[1:]]
                                ranker_triggered = True
                            else:
                                proposed = nominal
                        elif model == "contact_event_sequence":
                            contact_now = bool(
                                env.has_contact("tool_geom", "block_geom")
                                or env.has_contact("pusher_geom", "block_geom")
                                or env.last_has_contact("tool_geom", "block_geom")
                                or env.last_has_contact("pusher_geom", "block_geom")
                            )
                            if planned_sequence:
                                proposed = planned_sequence.pop(0)
                            elif not ranker_triggered and active and contact_now:
                                sequence = plan_push_calibrated(
                                    carrier_wrapped, selective, ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                    return_sequence=True,
                                )
                                proposed = sequence[0]
                                planned_sequence = [item for item in sequence[1:]]
                                ranker_triggered = True
                            else:
                                proposed = nominal
                        elif model == "carrier_residual_sequence":
                            contact_now = bool(
                                env.has_contact("tool_geom", "block_geom")
                                or env.has_contact("pusher_geom", "block_geom")
                                or env.last_has_contact("tool_geom", "block_geom")
                                or env.last_has_contact("pusher_geom", "block_geom")
                            )
                            if planned_sequence:
                                proposed = planned_sequence.pop(0)
                            elif not ranker_triggered and active and contact_now:
                                base_sequence = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                    return_sequence=True,
                                )
                                sequence = plan_push_calibrated(
                                    carrier_wrapped, selective, ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                    return_sequence=True, base_sequence=base_sequence,
                                )
                                proposed = sequence[0]
                                planned_sequence = [item for item in sequence[1:]]
                                ranker_triggered = True
                            else:
                                proposed = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                )
                        elif model == "carrier_selective_sequence":
                            contact_now = bool(
                                env.has_contact("tool_geom", "block_geom")
                                or env.has_contact("pusher_geom", "block_geom")
                                or env.last_has_contact("tool_geom", "block_geom")
                                or env.last_has_contact("pusher_geom", "block_geom")
                            )
                            if planned_sequence:
                                proposed = planned_sequence.pop(0)
                            elif not ranker_triggered and active and contact_now:
                                base_sequence = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                    return_sequence=True,
                                )
                                sequence = plan_push_calibrated(
                                    carrier_wrapped, selective, ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                    return_sequence=True, base_sequence=base_sequence,
                                    score_mode="selective",
                                )
                                proposed = sequence[0]
                                planned_sequence = [item for item in sequence[1:]]
                                ranker_triggered = True
                            else:
                                proposed = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                )
                        elif model == "carrier_neural_sequence":
                            contact_now = bool(
                                env.has_contact("tool_geom", "block_geom")
                                or env.has_contact("pusher_geom", "block_geom")
                                or env.last_has_contact("tool_geom", "block_geom")
                                or env.last_has_contact("pusher_geom", "block_geom")
                            )
                            if planned_sequence:
                                proposed = planned_sequence.pop(0)
                            elif not ranker_triggered and active and contact_now:
                                base_sequence = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                    return_sequence=True,
                                )
                                sequence = plan_push_neural_ranker(
                                    carrier_wrapped, selective, neural_ranker,
                                    observation["state"], nominal, target[:2], mask, angle,
                                    locked, candidates=args.candidates, horizon=args.horizon,
                                    seed=plan_seed, noise_std=args.ranker_noise_std,
                                    base_sequence=base_sequence,
                                    tool_block_distance=float(np.linalg.norm(env.ee_pos()[:2] - env.block_pos())),
                                )
                                proposed = sequence[0]
                                planned_sequence = [item for item in sequence[1:]]
                                ranker_triggered = True
                            else:
                                proposed = plan_push(
                                    carrier_wrapped, observation["state"], nominal, target[:2],
                                    mask, angle, env.joint_ranges, locked,
                                    candidates=args.candidates, horizon=args.horizon,
                                    iterations=args.iterations, seed=plan_seed,
                                )
                        else:
                            proposed = plan_push(
                                model, observation["state"], nominal, target[:2], mask, angle,
                                env.joint_ranges, locked, candidates=args.candidates,
                                horizon=args.horizon, iterations=args.iterations,
                                seed=plan_seed,
                            )
                        deviation = float(np.linalg.norm(proposed - nominal))
                        deviations.append(deviation)
                        if deviation <= args.guard_threshold:
                            action = proposed
                        else:
                            fallbacks += 1
                    observation = env.step(action)["observation"]
                    contacts += int(
                        env.has_contact("tool_geom", "block_geom")
                        or env.has_contact("pusher_geom", "block_geom")
                    )
                final = env.block_pos().copy()
                distance = float(np.linalg.norm(final - target[:2]))
                row = {
                    "seed": args.seed, "domain": domain.domain_id,
                    "target": target_item.target_id, "method": method,
                    "final_distance_m": distance,
                    "success_50mm": distance <= 0.05,
                    "block_displacement_m": float(np.linalg.norm(final - initial)),
                    "contact_steps": contacts, "fallback_steps": fallbacks,
                    "mean_action_deviation": float(np.mean(deviations)) if deviations else 0.0,
                }
                rows.append(row)
                print(row, flush=True)
    payload = {
        "version": "g2_ipwm_closed_loop_audit_v1",
        "frozen_protocol": {
            "approach_steps": args.approach_steps, "push_steps": args.push_steps,
            "candidates": args.candidates, "horizon": args.horizon,
            "iterations": args.iterations, "guard_threshold": args.guard_threshold,
            "ranker": None if args.ranker is None else str(args.ranker),
            "neural_ranker": None if args.neural_ranker is None else str(args.neural_ranker),
            "ranker_noise_std": args.ranker_noise_std,
            "ranker_contact_distance": args.ranker_contact_distance,
            "success_tolerance_m": 0.05,
        },
        "routing": routing, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
