"""Z48: causal episodic few-shot gate for BT-DPWM residual innovation.

Both BT-DPWM and the strongest shared h136/240 checkpoint receive the exact
same adapter, support trajectories, inner optimizer and outer update budget.
The adapter has no static bypass, so K=0 exactly recovers each frozen base.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.projected_residual_innovation import (
    FewShotProjectedModel, ProjectedResidualInnovation,
)
from robotarm.models.topology_graph_world_model import (
    TopologyGraphConfig, TopologyGraphWorldModel,
)
from robotarm.training.sim_protocol import DomainSpec, load_g1_protocol
from robotarm.training.g1_mechanism import residual_descriptor
from robotarm.envs.residual_physics import RESIDUAL_PROFILES, ResidualPhysicsConfig
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def rollout_loss(model, adapter, trajectories, domain, device, z, horizon,
                 topology_aware):
    states = torch.stack([x.states for x in trajectories]).to(device)
    actions = torch.stack([x.actions for x in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    model_mask, model_angle = ((mask, angle) if topology_aware else
                               (torch.zeros_like(mask), torch.zeros_like(angle)))
    free = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free.sum(-1).clamp_min(1.0)
    losses = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden, residual_hidden = states[:, start], None, None
        for offset in range(horizon):
            raw, hidden = model.step(prediction, actions[:, start + offset],
                                     model_mask, model_angle, hidden)
            correction, residual_hidden = adapter.step(
                prediction, actions[:, start + offset], mask, z, residual_hidden)
            prediction = raw + correction
            # Adapter is already zero on locked coordinates; projection is
            # repeated here to make the invariant explicit during inner loss.
            prediction[:, :5] = prediction[:, :5] * (1-mask) + angle * mask
            prediction[:, 5:10] = prediction[:, 5:10] * (1-mask)
        error = (prediction[:, :10] - states[:, start + horizon, :10]).pow(2)
        losses.append((error * free).sum(-1) / free_count)
    return torch.stack(losses).mean()


def infer_z(model, adapter, trajectories, domain, device, cfg, topology_aware):
    z = torch.zeros(adapter.latent_dim, device=device, requires_grad=True)
    for _ in range(int(cfg["inner_steps"])):
        loss = rollout_loss(model, adapter, trajectories, domain, device, z,
                            int(cfg["rollout_horizon"]), topology_aware)
        grad, = torch.autograd.grad(loss, z, create_graph=False)
        with torch.no_grad():
            z -= float(cfg["inner_learning_rate"]) * (
                grad + float(cfg["inner_l2"]) * z
            )
            z.clamp_(-3.0, 3.0)
        z.requires_grad_(True)
    return z.detach()


def batched_rollout_loss(model, adapter, trajectories, domains, device, z,
                         horizon, topology_aware, reduction="mean"):
    """One trajectory and one deployment latent per domain, in one GPU batch."""
    states = torch.stack([x.states for x in trajectories]).to(device)
    actions = torch.stack([x.actions for x in trajectories]).to(device)
    mask, angle = _damage_tensors([x.damage for x in domains], device)
    model_mask, model_angle = ((mask, angle) if topology_aware else
                               (torch.zeros_like(mask), torch.zeros_like(angle)))
    free = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free.sum(-1).clamp_min(1.0)
    losses = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden, residual_hidden = states[:, start], None, None
        for offset in range(horizon):
            raw, hidden = model.step(prediction, actions[:, start + offset],
                                     model_mask, model_angle, hidden)
            correction, residual_hidden = adapter.step(
                prediction, actions[:, start + offset], mask, z, residual_hidden)
            prediction = raw + correction
            prediction = torch.cat((
                prediction[:, :5] * (1-mask) + angle * mask,
                prediction[:, 5:10] * (1-mask), prediction[:, 10:]), -1)
        error = (prediction[:, :10] - states[:, start + horizon, :10]).pow(2)
        losses.append((error * free).sum(-1) / free_count)
    per_domain = torch.stack(losses).mean(0)
    if reduction == "none":
        return per_domain
    if reduction != "mean":
        raise ValueError(f"unknown reduction {reduction!r}")
    return per_domain.mean()


def infer_z_batched(model, adapter, trajectories, domains, device, cfg,
                    topology_aware):
    z = torch.zeros(len(domains), adapter.latent_dim, device=device,
                    requires_grad=True)
    for _ in range(int(cfg["inner_steps"])):
        loss = batched_rollout_loss(
            model, adapter, trajectories, domains, device, z,
            int(cfg["rollout_horizon"]), topology_aware)
        grad, = torch.autograd.grad(loss, z, create_graph=False)
        # batched loss is a mean; restore per-deployment gradient scale so the
        # inner learning rate is invariant to the number of parallel domains.
        grad = grad * len(domains)
        with torch.no_grad():
            z -= float(cfg["inner_learning_rate"]) * (
                grad + float(cfg["inner_l2"]) * z)
            z.clamp_(-3.0, 3.0)
        z.requires_grad_(True)
    return z.detach()


def train_adapter(model, adapter, domains, grouped, device, cfg, topology_aware,
                  seed, query_grouped=None, validation_grouped=None):
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(adapter.parameters(),
                                 lr=float(cfg["outer_learning_rate"]))
    rng = np.random.default_rng(seed + 48000)
    history = []
    best_state = copy.deepcopy(adapter.state_dict())
    best_validation = float("inf")
    non_regression_weight = float(cfg.get("non_regression_weight", 0.0))
    worst_domain_weight = float(cfg.get("worst_domain_weight", 0.0))
    for epoch in range(int(cfg["outer_epochs"])):
        budgets = cfg.get("training_transition_budgets")
        budget = (int(rng.choice(budgets)) if budgets else None)
        support, query = [], []
        for domain in domains:
            trajectories = grouped[domain.domain_id]
            indices = rng.permutation(len(trajectories))
            support_item = trajectories[int(indices[0])]
            if budget is not None:
                support_item = replace(
                    support_item,
                    states=support_item.states[:budget + 1],
                    actions=support_item.actions[:budget],
                    applied_actions=support_item.applied_actions[:budget],
                )
            support.append(support_item)
            query_trajectories = ((query_grouped or grouped)[domain.domain_id])
            query.append(query_trajectories[
                int(rng.integers(0, len(query_trajectories)))])
        if cfg.get("oracle_context_training", False):
            z = physical_context_batch(
                domains, device, query[0].states.dtype,
                centered=bool(cfg.get("center_physical_context", False)))
        else:
            z = infer_z_batched(model, adapter, support, domains, device, cfg,
                                topology_aware)
        per_domain = batched_rollout_loss(
            model, adapter, query, domains, device, z,
            int(cfg["rollout_horizon"]), topology_aware, reduction="none")
        with torch.no_grad():
            base_per_domain = batched_rollout_loss(
                model, adapter, query, domains, device,
                torch.zeros_like(z), int(cfg["rollout_horizon"]),
                topology_aware, reduction="none")
        regression = torch.relu(
            (per_domain - base_per_domain) / base_per_domain.clamp_min(1e-6))
        loss = (per_domain.mean() + non_regression_weight * regression.mean()
                + worst_domain_weight * regression.max())
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        mean = float(per_domain.mean().detach())
        entry = {"loss": mean, "objective": float(loss.detach()),
                 "regression_fraction": float(
                     (per_domain > base_per_domain).float().mean().detach()),
                 "support_transitions": budget}
        if validation_grouped is not None and (
                epoch == 0 or (epoch + 1) % int(cfg.get("validation_interval", 5)) == 0):
            validation = [validation_grouped[d.domain_id][0] for d in domains]
            validation_z = physical_context_batch(
                domains, device, validation[0].states.dtype,
                centered=bool(cfg.get("center_physical_context", False)))
            with torch.no_grad():
                val_per = batched_rollout_loss(
                    model, adapter, validation, domains, device, validation_z,
                    int(cfg["rollout_horizon"]), topology_aware, reduction="none")
                val_base = batched_rollout_loss(
                    model, adapter, validation, domains, device,
                    torch.zeros_like(validation_z), int(cfg["rollout_horizon"]),
                    topology_aware, reduction="none")
                val_reg = torch.relu(
                    (val_per-val_base)/val_base.clamp_min(1e-6))
                validation_score = float(
                    (val_per.mean()+non_regression_weight*val_reg.mean()
                     + worst_domain_weight*val_reg.max()).detach())
            entry["validation_objective"] = validation_score
            entry["validation_regression_fraction"] = float(
                (val_per > val_base).float().mean())
            if validation_score < best_validation:
                best_validation = validation_score
                best_state = copy.deepcopy(adapter.state_dict())
        history.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{cfg['outer_epochs']} query_mse={mean:.6f}",
                  flush=True)
    if validation_grouped is not None:
        adapter.load_state_dict(best_state)
    return history


def add_compositional_training_domains(protocol, cfg):
    """Register train-only physics combinations without exposing test profiles."""
    specifications = cfg.get("compositional_training_profiles", [])
    if not specifications:
        return protocol.train
    domains = list(protocol.train)
    held_out = {domain.residual_name for domain in protocol.test}
    for item in specifications:
        name = str(item["name"])
        if name in held_out:
            raise ValueError(f"training profile {name!r} leaks a held-out test profile")
        values = dict(item["physics"])
        if "actuator_scale" in values:
            values["actuator_scale"] = tuple(float(x) for x in values["actuator_scale"])
        if "backlash" in values:
            values["backlash"] = tuple(float(x) for x in values["backlash"])
        RESIDUAL_PROFILES[name] = ResidualPhysicsConfig(name=name, **values)
        domains.extend(DomainSpec(str(topology), name, "train")
                       for topology in item["topologies"])
    # The deployment assumption is known topology, unknown residual physics.
    # Optionally expose every test topology under train-only residual profiles
    # while keeping the held-out residual combinations themselves untouched.
    for topology in cfg.get("known_test_topologies", []):
        residual_names = tuple(dict.fromkeys(d.residual_name for d in domains))
        existing = {d.domain_id for d in domains}
        for residual_name in residual_names:
            candidate = DomainSpec(str(topology), residual_name, "train")
            if candidate.domain_id not in existing:
                domains.append(candidate)
                existing.add(candidate.domain_id)
    return tuple(domains)


def physical_context_batch(domains, device, dtype, *, centered=False):
    contexts = torch.stack([
        residual_descriptor(d.residual_name, device=device, dtype=dtype)
        for d in domains])
    if centered:
        contexts = contexts - contexts.mean(0, keepdim=True)
    return contexts


def cached_trajectories_by_domain(cache_dir, seed, suffix):
    """Load reusable per-domain trajectories from the newest compatible cache."""
    if not cache_dir:
        return {}
    candidates = sorted(Path(cache_dir).glob(f"seed{seed}_d*_{suffix}.pt"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {}
    grouped = {}
    for trajectory in torch.load(candidates[0], weights_only=False):
        grouped.setdefault(trajectory.domain_id, []).append(trajectory)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_fewshot_z48_v1.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-run", type=Path,
                        help="Optional frozen shared/BT checkpoint directory.")
    parser.add_argument("--bt-run", type=Path,
                        help="Optional BT checkpoint directory when distinct from shared.")
    parser.add_argument("--shared-adapter", type=Path,
                        help="Reuse a frozen fair shared adapter trained with this protocol.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(base_cfg["q0a_config"]).read_text(encoding="utf-8"))
    if args.smoke:
        cfg["outer_epochs"], cfg["inner_steps"] = 2, 2
        cfg["train_trajectories_per_domain"] = 2
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    checkpoint = (args.base_run or
                  Path("runs/g2_bt_dpwm_meta_train_z32") / f"seed{args.seed}_v1")
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(base_cfg["baseline_hidden_dim"]))).to(device)
    shared.load_state_dict(torch.load(checkpoint / "baseline_model.pt", map_location=device))
    bt = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(base_cfg["hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=True,
        object_hidden_dim=int(base_cfg["object_hidden_dim"]),
    ).to(device)
    bt_checkpoint = args.bt_run or checkpoint
    bt.load_state_dict(torch.load(bt_checkpoint / "model.pt", map_location=device))
    common_adapter = dict(
        latent_dim=8, rank=int(cfg["adapter_rank"]),
        hidden_dim=int(cfg["adapter_hidden_dim"]),
        position_limit=cfg.get("correction_position_limit"),
        velocity_limit=cfg.get("correction_velocity_limit"),
        factorized_context=bool(cfg.get("factorized_context", False)),
        joint_factorized_basis=bool(cfg.get("joint_factorized_basis", False)),
        memory_dim=int(cfg.get("adapter_memory_dim", 0)),
        analytic_history=bool(cfg.get("analytic_history", False)),
        history_deadband=float(cfg.get("history_deadband", 0.04)),
        shared_joint_basis=bool(cfg.get("shared_joint_basis", False)))
    torch.manual_seed(args.seed + 48000)
    shared_adapter = ProjectedResidualInnovation(**common_adapter).to(device)
    torch.manual_seed(args.seed + 48000)
    bt_adapter = ProjectedResidualInnovation(**common_adapter).to(device)

    train_domains = add_compositional_training_domains(protocol, cfg)
    print(f"[data] device={device} domains={len(train_domains)}", flush=True)
    train = collect_push_domains_warp(
        train_domains, trajectories_per_domain=int(cfg["train_trajectories_per_domain"]),
        steps=int(q0a["steps"]), seed=args.seed * 10000 + 480,
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float), excitation="active")
    count = int(cfg["train_trajectories_per_domain"])
    grouped = {d.domain_id: train[i*count:(i+1)*count]
               for i, d in enumerate(train_domains)}
    query_grouped = None
    if cfg.get("goal_query_training", False):
        query_count = int(cfg.get("goal_query_trajectories_per_domain", 4))
        cache_root = Path(cfg.get("goal_query_cache_dir", "runs/cache/bt_dpwm_goal_queries"))
        cache_root.mkdir(parents=True, exist_ok=True)
        domain_fingerprint = hashlib.sha256(
            "\n".join(d.domain_id for d in train_domains).encode()).hexdigest()[:10]
        query_cache = cache_root / (
            f"seed{args.seed}_d{len(train_domains)}_{domain_fingerprint}_q{query_count}.pt")
        if query_cache.exists():
            print(f"[data] loading goal query cache {query_cache}", flush=True)
            goal_queries = torch.load(query_cache, weights_only=False)
        else:
            reused = cached_trajectories_by_domain(
                cfg.get("goal_query_parent_cache_dir"), args.seed, f"q{query_count}")
            missing = tuple(d for d in train_domains if d.domain_id not in reused)
            print(f"[data] collecting {query_count} CPU goal queries/domain "
                  f"for {len(missing)} missing domains", flush=True)
            collected = collect_push_domains(
                missing, trajectories_per_domain=query_count,
                steps=int(q0a["steps"]), seed=args.seed * 10000 + 530,
                block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                excitation="goal",
                goal_exploration_std=float(q0a["goal_exploration_std"]),
                targets=tuple(x.as_array() for x in targets.calibration))
            collected_grouped = {
                d.domain_id: collected[i*query_count:(i+1)*query_count]
                for i, d in enumerate(missing)}
            goal_queries = [trajectory for d in train_domains for trajectory in
                            (reused.get(d.domain_id) or collected_grouped[d.domain_id])]
            torch.save(goal_queries, query_cache)
        query_grouped = {
            d.domain_id: goal_queries[i*query_count:(i+1)*query_count]
            for i, d in enumerate(train_domains)}
        validation_count = int(cfg.get("goal_validation_trajectories_per_domain", 1))
        validation_cache = cache_root / (
            f"seed{args.seed}_d{len(train_domains)}_{domain_fingerprint}_v{validation_count}.pt")
        if validation_cache.exists():
            goal_validation = torch.load(validation_cache, weights_only=False)
        else:
            reused_validation = cached_trajectories_by_domain(
                cfg.get("goal_query_parent_cache_dir"), args.seed,
                f"v{validation_count}")
            missing_validation = tuple(
                d for d in train_domains if d.domain_id not in reused_validation)
            collected_validation = collect_push_domains(
                missing_validation, trajectories_per_domain=validation_count,
                steps=int(q0a["steps"]), seed=args.seed * 10000 + 550,
                block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                excitation="goal",
                goal_exploration_std=float(q0a["goal_exploration_std"]),
                targets=tuple(x.as_array() for x in targets.validation))
            collected_validation_grouped = {
                d.domain_id: collected_validation[
                    i*validation_count:(i+1)*validation_count]
                for i, d in enumerate(missing_validation)}
            goal_validation = [trajectory for d in train_domains for trajectory in
                (reused_validation.get(d.domain_id)
                 or collected_validation_grouped[d.domain_id])]
            torch.save(goal_validation, validation_cache)
        validation_grouped = {
            d.domain_id: goal_validation[
                i*validation_count:(i+1)*validation_count]
            for i, d in enumerate(train_domains)}
    else:
        validation_grouped = None
    if args.shared_adapter:
        shared_adapter.load_state_dict(torch.load(
            args.shared_adapter, map_location=device))
        shared_history = [{"reused_from": str(args.shared_adapter)}]
        print(f"[train] reused shared fair adapter {args.shared_adapter}", flush=True)
    else:
        print("[train] shared fair adapter", flush=True)
        shared_history = train_adapter(shared, shared_adapter, train_domains, grouped,
                                       device, cfg, False, args.seed, query_grouped,
                                       validation_grouped)
    print("[train] BT-DPWM adapter", flush=True)
    bt_history = train_adapter(bt, bt_adapter, train_domains, grouped,
                               device, cfg, True, args.seed, query_grouped,
                               validation_grouped)

    output = args.output_dir or Path("runs/g2_bt_dpwm_fewshot_z48")/f"seed{args.seed}_v1"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(shared_adapter.state_dict(), output/"shared_adapter.pt")
    torch.save(bt_adapter.state_dict(), output/"bt_adapter.pt")
    if args.train_only:
        (output/"summary.json").write_text(json.dumps({
            "config_version": cfg["version"], "seed": args.seed,
            "smoke": args.smoke, "train_only": True,
            "shared_history": shared_history, "bt_history": bt_history,
        }, indent=2), encoding="utf-8")
        return

    rows = []
    common = dict(excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        goal_exploration_std=float(q0a["goal_exploration_std"]))
    for index, domain in enumerate(protocol.test):
        calibration = collect_push_domains(
            (domain,), trajectories_per_domain=max(cfg["calibration_shots"]),
            steps=int(q0a["steps"]), seed=args.seed*100000+index*1000+100,
            targets=tuple(x.as_array() for x in targets.calibration), **common)
        test = collect_push_domains(
            (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            steps=int(q0a["steps"]), seed=args.seed*100000+index*1000+500,
            targets=tuple(x.as_array() for x in targets.evaluation), **common)
        for k in cfg["calibration_shots"]:
            if k == 0:
                shared_z = torch.zeros(8, device=device); bt_z = shared_z.clone()
            else:
                shared_z = infer_z(shared, shared_adapter, calibration[:k], domain,
                                   device, cfg, False)
                bt_z = infer_z(bt, bt_adapter, calibration[:k], domain,
                               device, cfg, True)
            shared_wrapped = FewShotProjectedModel(
                shared, shared_adapter, base_uses_topology=False).to(device)
            bt_wrapped = FewShotProjectedModel(bt, bt_adapter).to(device)
            shared_wrapped.set_residual_context(shared_z)
            bt_wrapped.set_residual_context(bt_z)
            result = evaluate({"shared": shared_wrapped, "bt_dpwm": bt_wrapped},
                              domain, test, device, int(q0a["rollout_horizon"]),
                              topology_aware_methods=("shared", "bt_dpwm"))
            values = {x["method"]: x for x in result}
            base, candidate = values["shared"], values["bt_dpwm"]
            improve = 100*(base["overall_rmse"]-candidate["overall_rmse"])/base["overall_rmse"]
            rows.append({"domain": domain.domain_id, "k": k,
                         "improvement_pct": improve, "shared": base,
                         "bt_dpwm": candidate, "shared_z_norm": float(shared_z.norm()),
                         "bt_z_norm": float(bt_z.norm())})
            print(f"[eval] {domain.domain_id} K={k} improvement={improve:+.2f}%", flush=True)
    summary = {"config_version": cfg["version"], "seed": args.seed,
               "smoke": args.smoke, "shared_history": shared_history,
               "bt_history": bt_history, "rows": rows}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
