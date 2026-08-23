"""Y0: fair single-model gate for the first executable BT-DPWM."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_dual_expert_fair_gate_v0 import _component_loss, train_model
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_object_preserving_projection_x1 import _losses
from scripts.run_push_benchmark import collect_push_domains


def cached_collect(cache_dir, cache_key, collector):
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    path = cache_dir / f"{digest}.pt"
    if path.exists():
        print(f"[cache hit] {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    print(f"[cache miss] {path}", flush=True)
    trajectories = collector()
    torch.save(trajectories, path)
    return trajectories


def robot_losses_per_trajectory(model, batch, horizon, use_topology=False):
    states, actions, mask, angle = batch
    zeros = torch.zeros_like(mask)
    model_mask, model_angle = (mask, angle) if use_topology else (zeros, zeros)
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    one_step, hidden = [], None
    for step in range(actions.shape[1]):
        robot, hidden, _, _, _ = model.step_robot(
            states[:, step], actions[:, step], model_mask, model_angle, hidden)
        error = (robot - states[:, step + 1, :10]).pow(2)
        one_step.append((error * free_mask).sum(-1) / free_count)
    rollout = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden = states[:, start], None
        for offset in range(horizon):
            robot, hidden, obj, _, _ = model.step_robot(
                prediction, actions[:, start + offset], model_mask, model_angle, hidden)
            error = (robot - states[:, start + offset + 1, :10]).pow(2)
            rollout.append((error * free_mask).sum(-1) / free_count)
            prediction = torch.cat((robot, obj), -1)
    return torch.stack(one_step).mean(0) + 0.5 * torch.stack(rollout).mean(0)


def robot_losses(model, batch, horizon, use_topology=False):
    return robot_losses_per_trajectory(model, batch, horizon, use_topology).mean()


def train_robot_only(model, batch, *, epochs, learning_rate, horizon, use_topology=False):
    parameters = [p for name, p in model.named_parameters()
                  if name.startswith("robot_") or name.startswith("additional_robot_experts.")]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    for epoch in range(epochs):
        loss = robot_losses(model, batch, horizon, use_topology)
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch+1:03d} loss={loss.item():.6f} grad={gradient:.3f}", flush=True)
    return history


def train_robot_only_with_selection(
    model, batch, validation_batch, *, epochs, learning_rate, horizon,
    validation_every, train_group_indices, validation_group_indices,
    group_robust_weight, use_topology=False,
):
    """Robot-only group-robust training selected without consulting test domains."""
    parameters = [p for name, p in model.named_parameters()
                  if name.startswith("robot_") or name.startswith("additional_robot_experts.")]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history, validation_history = [], []

    @torch.no_grad()
    def validation_value():
        losses = robot_losses_per_trajectory(
            model, validation_batch, horizon, use_topology
        )
        value, groups = aggregate_topology_losses(
            losses, validation_group_indices, group_robust_weight
        )
        return float(value), {name: float(loss) for name, loss in groups.items()}

    best_value, best_groups = validation_value()
    best_epoch = 0
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    validation_history.append({
        "epoch": 0, "loss": best_value, "topology_losses": best_groups,
    })
    for epoch in range(1, epochs + 1):
        losses = robot_losses_per_trajectory(model, batch, horizon, use_topology)
        loss, _ = aggregate_topology_losses(
            losses, train_group_indices, group_robust_weight
        )
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        suffix = ""
        if epoch % validation_every == 0 or epoch == epochs:
            value, groups = validation_value()
            validation_history.append({
                "epoch": epoch, "loss": value, "topology_losses": groups,
            })
            if value < best_value:
                best_value, best_groups, best_epoch = value, groups, epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            suffix = f" val={value:.6f}"
        if epoch == 1 or epoch % 10 == 0:
            print(f"  robot-select epoch={epoch:03d} loss={loss.item():.6f}{suffix} "
                  f"grad={gradient:.3f}", flush=True)
    model.load_state_dict(best_state)
    diagnostics = {
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_value,
        "selected_topology_losses": best_groups,
        "validation_history": validation_history,
        "group_robust_weight": group_robust_weight,
    }
    print(f"[robot] selected epoch={best_epoch} validation_loss={best_value:.6f}", flush=True)
    return history, diagnostics


def object_losses_per_trajectory(model, batch, horizon, use_topology=False):
    states, actions, mask, angle = batch
    zeros = torch.zeros_like(mask)
    model_mask, model_angle = (mask, angle) if use_topology else (zeros, zeros)
    one_step, hidden = [], None
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], model_mask, model_angle, hidden
        )
        one_step.append(
            (prediction[:, 10:] - states[:, step + 1, 10:]).pow(2).mean(-1)
        )
    rollout = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden = states[:, start], None
        for offset in range(horizon):
            prediction, hidden = model.step(
                prediction, actions[:, start + offset], model_mask, model_angle, hidden
            )
            rollout.append(
                (prediction[:, 10:] - states[:, start + offset + 1, 10:]).pow(2).mean(-1)
            )
    return torch.stack(one_step).mean(0) + 0.5 * torch.stack(rollout).mean(0)


def train_object_with_selection(
    model, batch, validation_batch, *, epochs, learning_rate, horizon,
    validation_every, train_group_indices, validation_group_indices,
    group_robust_weight, use_topology=False,
):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history, validation_history = [], []
    with torch.no_grad():
        initial_values = object_losses_per_trajectory(
            model, validation_batch, horizon, use_topology)
        initial_value, initial_groups = aggregate_topology_losses(
            initial_values, validation_group_indices, group_robust_weight)
    best_value, best_epoch = float(initial_value), 0
    best_state = {name: tensor.detach().clone()
                  for name, tensor in model.state_dict().items()}
    validation_history.append({
        "epoch": 0, "loss": best_value,
        "topology_losses": {name: float(item) for name, item in initial_groups.items()},
    })
    for epoch in range(1, epochs + 1):
        losses = object_losses_per_trajectory(model, batch, horizon, use_topology)
        loss, _ = aggregate_topology_losses(
            losses, train_group_indices, group_robust_weight
        )
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        suffix = ""
        if epoch % validation_every == 0 or epoch == epochs:
            with torch.no_grad():
                values = object_losses_per_trajectory(
                    model, validation_batch, horizon, use_topology
                )
                value, groups = aggregate_topology_losses(
                    values, validation_group_indices, group_robust_weight
                )
            value_float = float(value)
            group_values = {name: float(item) for name, item in groups.items()}
            validation_history.append({"epoch": epoch, "loss": value_float,
                                       "topology_losses": group_values})
            if value_float < best_value:
                best_value, best_epoch = value_float, epoch
                best_state = {name: tensor.detach().clone()
                              for name, tensor in model.state_dict().items()}
            suffix = f" val={value_float:.6f}"
        if epoch == 1 or epoch % 10 == 0:
            print(f"  object-select epoch={epoch:03d} loss={loss.item():.6f}{suffix} "
                  f"grad={gradient:.3f}", flush=True)
    model.load_state_dict(best_state)
    diagnostics = {"selected_epoch": best_epoch, "selected_validation_loss": best_value,
                   "validation_history": validation_history,
                   "group_robust_weight": group_robust_weight}
    print(f"[object] selected epoch={best_epoch} validation_loss={best_value:.6f}", flush=True)
    return history, diagnostics


def train_blockwise_horizons(model, batch, *, epochs, learning_rate,
                             robot_horizon, object_horizon):
    """Optimize each directed block at its empirically identified time scale."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        joint, _ = _losses(model, batch, robot_horizon)
        _, obj = _losses(model, batch, object_horizon)
        loss = joint + obj
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch+1:03d} loss={loss.item():.6f} joint={joint.item():.6f} "
                  f"object={obj.item():.6f} grad={gradient:.3f}", flush=True)
    return history


def shared_losses_per_trajectory(model, batch, horizon, use_topology=False):
    """Return the shared rollout objective separately for every trajectory."""
    states, actions, mask, angle = batch
    zeros = torch.zeros_like(mask)
    model_mask, model_angle = (mask, angle) if use_topology else (zeros, zeros)
    hidden, one_step = None, []
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], model_mask, model_angle, hidden
        )
        target = states[:, step + 1]
        one_step.append(
            (prediction[:, :10] - target[:, :10]).pow(2).mean(-1)
            + (prediction[:, 10:] - target[:, 10:]).pow(2).mean(-1)
        )
    rollout = []
    rollout_horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - rollout_horizon + 1, rollout_horizon):
        prediction, hidden = states[:, start], None
        for offset in range(rollout_horizon):
            prediction, hidden = model.step(
                prediction, actions[:, start + offset], model_mask, model_angle, hidden
            )
            target = states[:, start + offset + 1]
            rollout.append(
                (prediction[:, :10] - target[:, :10]).pow(2).mean(-1)
                + (prediction[:, 10:] - target[:, 10:]).pow(2).mean(-1)
            )
    return torch.stack(one_step).mean(0) + 0.5 * torch.stack(rollout).mean(0)


def topology_group_indices(trajectories, device):
    """Group trajectory rows by damage topology, never by residual/test label."""
    topologies = [item.domain_id.split("__", 1)[0] for item in trajectories]
    return {
        topology: torch.tensor(
            [index for index, value in enumerate(topologies) if value == topology],
            device=device, dtype=torch.long,
        )
        for topology in sorted(set(topologies))
    }


def aggregate_topology_losses(per_trajectory, group_indices, robust_weight):
    """Convex average/worst-topology objective and auditable group losses."""
    if not 0.0 <= robust_weight <= 1.0:
        raise ValueError("group robust weight must be in [0, 1]")
    group_losses = {
        name: per_trajectory.index_select(0, indices).mean()
        for name, indices in group_indices.items()
    }
    stacked = torch.stack(list(group_losses.values()))
    objective = (1.0 - robust_weight) * stacked.mean() + robust_weight * stacked.max()
    return objective, group_losses


@torch.no_grad()
def shared_loss(model, batch, horizon, use_topology=False):
    """Deterministic mean rollout objective used by legacy frozen selections."""
    return shared_losses_per_trajectory(model, batch, horizon, use_topology).mean()


@torch.no_grad()
def robust_validation_loss(model, batch, group_indices, horizon, use_topology,
                           robust_weight):
    per_trajectory = shared_losses_per_trajectory(model, batch, horizon, use_topology)
    value, groups = aggregate_topology_losses(
        per_trajectory, group_indices, robust_weight
    )
    return float(value), {name: float(loss) for name, loss in groups.items()}


def train_shared_with_selection(model, batch, validation_batch, *, epochs,
                                learning_rate, horizon, validation_every,
                                ema_decay, use_topology=False,
                                train_group_indices=None,
                                validation_group_indices=None,
                                group_robust_weight=0.0):
    """Train once and select final/EMA/checkpoint only on the frozen validation split."""
    states, actions, mask, angle = batch
    zeros = torch.zeros_like(mask)
    model_mask, model_angle = (mask, angle) if use_topology else (zeros, zeros)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, validation_history = [], []
    ema = None
    best_state, best_value, best_epoch = None, float("inf"), None
    for epoch in range(1, epochs + 1):
        per_trajectory = shared_losses_per_trajectory(
            model, (states, actions, model_mask, model_angle), horizon, use_topology=True
        )
        if train_group_indices is None:
            loss = per_trajectory.mean()
        else:
            loss, _ = aggregate_topology_losses(
                per_trajectory, train_group_indices, group_robust_weight
            )
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        state = model.state_dict()
        if ema is None:
            ema = {name: value.detach().clone() for name, value in state.items()}
        else:
            for name, value in state.items():
                ema[name].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        if epoch % validation_every == 0 or epoch == epochs:
            if validation_group_indices is None:
                value = float(shared_loss(
                    model, validation_batch, horizon, use_topology=use_topology
                )); group_values = None
            else:
                value, group_values = robust_validation_loss(
                    model, validation_batch, validation_group_indices, horizon,
                    use_topology, group_robust_weight,
                )
            record = {"epoch": epoch, "loss": value}
            if group_values is not None:
                record["topology_losses"] = group_values
            validation_history.append(record)
            if value < best_value:
                best_state = {name: tensor.detach().clone() for name, tensor in state.items()}
                best_value, best_epoch = value, epoch
        if epoch == 1 or epoch % 10 == 0:
            suffix = f" val={value:.6f}" if (epoch % validation_every == 0 or epoch == epochs) else ""
            print(f"  select epoch={epoch:03d} loss={loss.item():.6f}{suffix} "
                  f"grad={gradient:.3f}", flush=True)
    final_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    validation_fn = lambda: (
        robust_validation_loss(model, validation_batch, validation_group_indices,
                               horizon, use_topology, group_robust_weight)[0]
        if validation_group_indices is not None else
        float(shared_loss(model, validation_batch, horizon, use_topology=use_topology))
    )
    final_value = validation_fn()
    model.load_state_dict(ema)
    ema_value = validation_fn()
    candidates = {
        "final": (final_value, final_state, epochs),
        "ema": (ema_value, ema, epochs),
        "validation_best": (best_value, best_state, best_epoch),
    }
    selected_name, (selected_value, selected_state, selected_epoch) = min(
        candidates.items(), key=lambda item: item[1][0]
    )
    model.load_state_dict(selected_state)
    diagnostics = {
        "validation_history": validation_history,
        "candidate_validation_loss": {name: value[0] for name, value in candidates.items()},
        "selected": selected_name,
        "selected_epoch": selected_epoch,
        "selected_validation_loss": selected_value,
        "ema_decay": ema_decay,
        "group_robust_weight": group_robust_weight,
    }
    print(f"[scaffold] selected={selected_name} epoch={selected_epoch} "
          f"validation_loss={selected_value:.6f}", flush=True)
    return history, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v0-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in cfg["seeds"]:
        raise ValueError("seed not in frozen Y0 list")
    v0 = yaml.safe_load(Path(cfg["v0_config"]).read_text(encoding="utf-8"))
    q0a_path = Path(cfg.get("q0a_config", v0["q0a_config"]))
    q0a = yaml.safe_load(q0a_path.read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    train_key = json.dumps({"kind": "push_train", "seed": args.seed,
        "domains": [x.domain_id for x in protocol.train], "q0a": q0a}, sort_keys=True)
    train_data = cached_collect(args.cache_dir, train_key, lambda: collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common))
    model_cfg = TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))
    baseline_cfg = TopologyGraphConfig(hidden_dim=int(cfg.get("baseline_hidden_dim", cfg["hidden_dim"])))
    scaffold_cfg = TopologyGraphConfig(
        hidden_dim=int(cfg.get("baseline_hidden_dim", cfg["hidden_dim"])),
        contact_gated_object_context=bool(cfg.get("contact_gated_object_context", False)),
        contact_gate_threshold=float(cfg.get("reaction_gate_threshold", -0.005)),
        contact_gate_temperature=float(cfg.get("reaction_gate_temperature", 0.002)),
        kinematic_integration_dt=cfg.get("kinematic_integration_dt"),
        kinematic_position_blend=float(cfg.get("kinematic_position_blend", 1.0)),
    )
    baseline = TopologyGraphWorldModel(baseline_cfg).to(device)
    if bool(cfg.get("train_baseline", False)):
        torch.manual_seed(args.seed)
        baseline = TopologyGraphWorldModel(model_cfg).to(device)
        print("[baseline] train shared compute-matched", flush=True)
        baseline_history = train_model(
            baseline, _batch(train_data, device), component="shared",
            epochs=int(cfg["baseline_epochs"]), learning_rate=float(cfg["learning_rate"]),
            rollout_horizon=int(cfg["baseline_rollout_training_horizon"]),
        )
    elif "external_baseline_model_template" in cfg:
        baseline_path = Path(str(cfg["external_baseline_model_template"]).format(seed=args.seed))
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
        baseline_history = None
        print(f"[baseline] loaded {baseline_path}", flush=True)
    else:
        baseline.load_state_dict(torch.load(args.v0_run_dir / "models.pt", map_location=device)["shared_compute_matched"])
        baseline_history = None
    scaffold_source = baseline
    validation_data = None
    swa_history, swa_count, scaffold_selection = None, 0, None
    if int(cfg.get("selection_scaffold_epochs", 0)):
        validation_key = json.dumps({"kind": "push_validation", "seed": args.seed,
            "domains": [x.domain_id for x in protocol.validation], "q0a": q0a}, sort_keys=True)
        validation_data = cached_collect(args.cache_dir, validation_key, lambda: collect_push_domains(
            protocol.validation,
            trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            seed=args.seed * 10_000 + 750,
            targets=tuple(x.as_array() for x in targets.validation), **common))
        torch.manual_seed(args.seed)
        scaffold_source = TopologyGraphWorldModel(scaffold_cfg).to(device)
        print("[scaffold] train shared selection source", flush=True)
        robust_weight = float(cfg.get("selection_group_robust_weight", 0.0))
        train_groups = (
            topology_group_indices(train_data, device) if robust_weight > 0.0 else None
        )
        validation_groups = (
            topology_group_indices(validation_data, device) if robust_weight > 0.0 else None
        )
        if train_groups is not None:
            print(f"[scaffold] topology robust weight={robust_weight:.2f} "
                  f"groups={list(train_groups)}", flush=True)
        swa_history, scaffold_selection = train_shared_with_selection(
            scaffold_source, _batch(train_data, device), _batch(validation_data, device),
            epochs=int(cfg["selection_scaffold_epochs"]),
            learning_rate=float(cfg["selection_scaffold_learning_rate"]),
            horizon=int(cfg["selection_scaffold_rollout_horizon"]),
            validation_every=int(cfg.get("selection_validation_every", 10)),
            ema_decay=float(cfg.get("selection_ema_decay", 0.99)),
            use_topology=bool(cfg.get("selection_use_topology", False)),
            train_group_indices=train_groups,
            validation_group_indices=validation_groups,
            group_robust_weight=robust_weight,
        )
    if int(cfg.get("swa_scaffold_epochs", 0)):
        torch.manual_seed(args.seed)
        scaffold_source = TopologyGraphWorldModel(baseline_cfg).to(device)
        print("[scaffold] train shared SWA source", flush=True)
        swa_history, swa_count = train_shared_with_swa(
            scaffold_source, _batch(train_data, device),
            epochs=int(cfg["swa_scaffold_epochs"]),
            learning_rate=float(cfg["swa_scaffold_learning_rate"]),
            horizon=int(cfg["swa_scaffold_rollout_horizon"]),
            swa_start=int(cfg["swa_scaffold_start"]),
        )
    torch.manual_seed(args.seed)
    candidate = BlockTriangularDPWM(
        model_cfg,
        contact_conditioned_robot=bool(cfg.get("contact_conditioned_robot", False)),
        independent_object_encoder=bool(cfg.get("independent_object_encoder", False)),
        object_hidden_dim=int(cfg.get("object_hidden_dim", cfg["hidden_dim"])),
        reaction_rank=int(cfg.get("reaction_rank", 0)),
        reaction_geometry_gate=bool(cfg.get("reaction_geometry_gate", False)),
        reaction_gate_threshold=float(cfg.get("reaction_gate_threshold", -0.005)),
        reaction_gate_temperature=float(cfg.get("reaction_gate_temperature", 0.002)),
        reaction_scale=float(cfg.get("reaction_scale", 1.0)),
        reaction_physical_features=bool(cfg.get("reaction_physical_features", False)),
        reaction_event_decay=cfg.get("reaction_event_decay"),
        reaction_fixed_initialization=bool(cfg.get("reaction_fixed_initialization", False)),
        kinematic_integration_dt=cfg.get("kinematic_integration_dt"),
        kinematic_position_blend=float(cfg.get("kinematic_position_blend", 1.0)),
        shadow_object_rank=int(cfg.get("shadow_object_rank", 0)),
        robot_expert_count=int(cfg.get("robot_expert_count", 1)),
        contact_gated_object_context=bool(cfg.get("contact_gated_object_context", False)),
        compact_bridge_object_head=bool(cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(cfg.get("geometric_object_rank", 0)),
    ).to(device)
    if "initialize_candidate_full_template" in cfg:
        source_path = Path(str(cfg["initialize_candidate_full_template"]).format(seed=args.seed))
        missing, unexpected = candidate.load_state_dict(
            torch.load(source_path, map_location=device), strict=False)
        allowed_missing = {name for name in candidate.state_dict()
                           if name.startswith("geometric_object_head.")}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(f"incompatible full template: missing={missing}, unexpected={unexpected}")
        print(f"[initialize] loaded frozen BT template {source_path}; "
              f"new geometry parameters={sum(candidate.state_dict()[x].numel() for x in missing):,}",
              flush=True)
    candidate_template_robot = None
    if "initialize_candidate_model_template" in cfg:
        source_path = Path(str(cfg["initialize_candidate_model_template"]).format(seed=args.seed))
        source = torch.load(source_path, map_location=device)
        current = candidate.state_dict()
        compatible = {name: value for name, value in source.items()
                      if (name.startswith("robot_") or name.startswith("additional_robot_experts."))
                      and name in current and current[name].shape == value.shape}
        candidate.load_state_dict({**current, **compatible})
        candidate_template_robot = {
            name: value.detach().clone() for name, value in compatible.items()
        }
        print(f"[initialize] loaded {sum(x.numel() for x in compatible.values()):,} compatible "
              f"parameters from {source_path}", flush=True)
    if bool(cfg.get("initialize_robot_from_baseline", False)):
        source = scaffold_source.state_dict(); target = candidate.state_dict()
        prefixes = {
            "node_encoder.": "robot_encoder.",
            "message.": "robot_message.",
            "update.": "robot_update.",
            "temporal.": "robot_temporal.",
            "joint_head.": "robot_head.",
        }
        copied = 0
        for source_prefix, target_prefix in prefixes.items():
            for name, value in source.items():
                if name.startswith(source_prefix):
                    destination = target_prefix + name[len(source_prefix):]
                    if destination in target and target[destination].shape == value.shape:
                        target[destination] = value.detach().clone(); copied += value.numel()
        candidate.load_state_dict(target)
        print(f"[initialize] copied {copied:,} robot parameters from baseline", flush=True)
    if bool(cfg.get("zero_topology_input_weights", False)):
        with torch.no_grad():
            candidate.robot_encoder[0].weight[:, 3:5].zero_()
        print("[initialize] zeroed untrained topology input columns; projection remains active",
              flush=True)
    if bool(cfg.get("initialize_object_from_baseline", False)):
        source = baseline.state_dict(); target = candidate.state_dict()
        copied = 0
        for name, value in source.items():
            if (name.startswith("object_head.") and name in target
                    and target[name].shape == value.shape):
                target[name] = value.detach().clone(); copied += value.numel()
        candidate.load_state_dict(target)
        print(f"[initialize] copied {copied:,} object parameters from baseline", flush=True)
    blend_selection = None
    if bool(cfg.get("select_robot_weight_blend", False)):
        if candidate_template_robot is None:
            raise ValueError("robot weight blend requires initialize_candidate_model_template")
        if validation_data is None:
            validation_key = json.dumps({"kind": "push_validation", "seed": args.seed,
                "domains": [x.domain_id for x in protocol.validation], "q0a": q0a},
                sort_keys=True)
            validation_data = cached_collect(
                args.cache_dir, validation_key, lambda: collect_push_domains(
                    protocol.validation,
                    trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                    seed=args.seed * 10_000 + 750,
                    targets=tuple(x.as_array() for x in targets.validation), **common,
                )
            )
        baseline_state = baseline.state_dict()
        current = candidate.state_dict()
        prefixes = {
            "node_encoder.": "robot_encoder.", "message.": "robot_message.",
            "update.": "robot_update.", "temporal.": "robot_temporal.",
            "joint_head.": "robot_head.",
        }
        projected_shared = {}
        for source_prefix, target_prefix in prefixes.items():
            for name, value in baseline_state.items():
                if name.startswith(source_prefix):
                    destination = target_prefix + name[len(source_prefix):]
                    if destination in current and current[destination].shape == value.shape:
                        projected_shared[destination] = value.detach().clone()
        projected_shared["robot_encoder.0.weight"][:, 3:5].zero_()
        validation_batch = _batch(validation_data, device)
        validation_groups = topology_group_indices(validation_data, device)
        robust_weight = float(cfg.get("robot_blend_group_robust_weight", 0.5))
        candidates = []
        for alpha in [float(value) for value in cfg["robot_blend_alphas"]]:
            mixed = {
                name: projected_shared[name].lerp(candidate_template_robot[name], alpha)
                for name in projected_shared
            }
            candidate.load_state_dict({**current, **mixed})
            with torch.no_grad():
                losses = robot_losses_per_trajectory(
                    candidate, validation_batch,
                    int(cfg["robot_rollout_training_horizon"]), True,
                )
                value, groups = aggregate_topology_losses(
                    losses, validation_groups, robust_weight
                )
            candidates.append({
                "alpha": alpha, "loss": float(value),
                "topology_losses": {name: float(loss) for name, loss in groups.items()},
                "state": mixed,
            })
        selected = min(candidates, key=lambda item: item["loss"])
        candidate.load_state_dict({**current, **selected["state"]})
        blend_selection = {
            "selected_alpha": selected["alpha"],
            "selected_validation_loss": selected["loss"],
            "group_robust_weight": robust_weight,
            "candidates": [{key: value for key, value in item.items() if key != "state"}
                           for item in candidates],
        }
        print(f"[initialize] selected robot weight blend alpha={selected['alpha']:.2f} "
              f"validation_loss={selected['loss']:.6f}", flush=True)
    topology_hook = None
    if bool(cfg.get("topology_input_only_training", False)):
        if not bool(cfg.get("internal_topology_conditioning", False)):
            raise ValueError("topology_input_only_training requires internal topology conditioning")
        first_weight = candidate.robot_encoder[0].weight
        with torch.no_grad():
            first_weight[:, 3:5].zero_()
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        first_weight.requires_grad_(True)
        gradient_mask = torch.zeros_like(first_weight)
        gradient_mask[:, 3:5] = 1.0
        topology_hook = first_weight.register_hook(lambda gradient: gradient * gradient_mask)
        print(f"[initialize] topology-input adapter trains {gradient_mask.sum().item():.0f} existing weights",
              flush=True)
    if bool(cfg.get("robot_head_only_training", False)):
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(name.startswith("robot_head."))
        trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        print(f"[initialize] robot-head adaptation trains {trainable:,} existing weights", flush=True)
    batch = _batch(train_data, device)
    use_topology = bool(cfg.get("internal_topology_conditioning", False))
    robot_selection = None
    object_selection = None
    refinement_epochs = int(cfg.get("joint_refinement_epochs", 0))
    shared_epochs = int(cfg["epochs"]) - refinement_epochs
    if bool(cfg.get("block_coordinate_training", False)):
        for name, parameter in candidate.named_parameters():
            if name.startswith("object_"):
                parameter.requires_grad_(False)
        print("[block 1/2] robot", flush=True)
        if int(cfg["robot_epochs"]) == 0:
            robot_history = []
            print("[block 1/2] frozen pretrained robot; no additional updates", flush=True)
        elif bool(cfg.get("robot_only_forward", False)):
            if bool(cfg.get("robot_validation_selection", False)):
                if validation_data is None:
                    validation_key = json.dumps({"kind": "push_validation", "seed": args.seed,
                        "domains": [x.domain_id for x in protocol.validation], "q0a": q0a},
                        sort_keys=True)
                    validation_data = cached_collect(
                        args.cache_dir, validation_key, lambda: collect_push_domains(
                            protocol.validation,
                            trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                            seed=args.seed * 10_000 + 750,
                            targets=tuple(x.as_array() for x in targets.validation), **common,
                        )
                    )
                robust_weight = float(cfg.get("robot_group_robust_weight", 0.5))
                robot_history, robot_selection = train_robot_only_with_selection(
                    candidate, batch, _batch(validation_data, device),
                    epochs=int(cfg["robot_epochs"]),
                    learning_rate=float(cfg.get("robot_learning_rate", cfg["learning_rate"])),
                    horizon=int(cfg["robot_rollout_training_horizon"]),
                    validation_every=int(cfg.get("robot_validation_every", 5)),
                    train_group_indices=topology_group_indices(train_data, device),
                    validation_group_indices=topology_group_indices(validation_data, device),
                    group_robust_weight=robust_weight, use_topology=use_topology,
                )
            else:
                robot_history = train_robot_only(
                    candidate, batch, epochs=int(cfg["robot_epochs"]),
                    learning_rate=float(cfg.get("robot_learning_rate", cfg["learning_rate"])),
                    horizon=int(cfg["robot_rollout_training_horizon"]),
                    use_topology=use_topology,
                )
        else:
            robot_history = train_model(
                candidate, batch, component="joint", epochs=int(cfg["robot_epochs"]),
                learning_rate=float(cfg.get("robot_learning_rate", cfg["learning_rate"])),
                rollout_horizon=int(cfg["robot_rollout_training_horizon"]),
                use_topology=use_topology,
            )
        if topology_hook is not None:
            topology_hook.remove()
        shadow_epochs = int(cfg.get("shadow_epochs", 0))
        shadow_history = []
        if shadow_epochs:
            for name, parameter in candidate.named_parameters():
                parameter.requires_grad_(name.startswith("shadow_context_head."))
            print(f"[block 2/3] shadow context epochs={shadow_epochs}", flush=True)
            shadow_history = train_model(
                candidate, batch, component="joint", epochs=shadow_epochs,
                learning_rate=float(cfg.get("shadow_learning_rate", cfg["learning_rate"])),
                rollout_horizon=int(cfg["robot_rollout_training_horizon"]),
                use_topology=use_topology,
            )
        geometric_only = bool(cfg.get("geometric_object_only_training", False))
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(
                name.startswith("geometric_object_head.") if geometric_only
                else name.startswith("object_"))
        print("[block object] object on frozen robot/shadow rollouts", flush=True)
        if bool(cfg.get("object_validation_selection", False)):
            if validation_data is None:
                validation_key = json.dumps({"kind": "push_validation", "seed": args.seed,
                    "domains": [x.domain_id for x in protocol.validation], "q0a": q0a},
                    sort_keys=True)
                validation_data = cached_collect(
                    args.cache_dir, validation_key, lambda: collect_push_domains(
                        protocol.validation,
                        trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                        seed=args.seed * 10_000 + 750,
                        targets=tuple(x.as_array() for x in targets.validation), **common,
                    )
                )
            object_history, object_selection = train_object_with_selection(
                candidate, batch, _batch(validation_data, device),
                epochs=int(cfg["object_epochs"]),
                learning_rate=float(cfg.get("object_learning_rate", cfg["learning_rate"])),
                horizon=int(cfg["object_rollout_training_horizon"]),
                validation_every=int(cfg.get("object_validation_every", 5)),
                train_group_indices=topology_group_indices(train_data, device),
                validation_group_indices=topology_group_indices(validation_data, device),
                group_robust_weight=float(cfg.get("object_group_robust_weight", 0.5)),
                use_topology=use_topology,
            )
        else:
            object_history = train_model(
                candidate, batch, component="object", epochs=int(cfg["object_epochs"]),
                learning_rate=float(cfg.get("object_learning_rate", cfg["learning_rate"])),
                rollout_horizon=int(cfg["object_rollout_training_horizon"]),
                use_topology=use_topology,
            )
        history = robot_history + shadow_history + object_history
        reaction_epochs = int(cfg.get("reaction_epochs", 0))
        if reaction_epochs:
            for name, parameter in candidate.named_parameters():
                parameter.requires_grad_(name.startswith("reaction_adapter."))
            print(f"[block 3/3] reaction adapter epochs={reaction_epochs}", flush=True)
            reaction_history = train_model(
                candidate, batch, component="joint", epochs=reaction_epochs,
                learning_rate=float(cfg["reaction_learning_rate"]),
                rollout_horizon=int(cfg["robot_rollout_training_horizon"]),
                use_topology=use_topology,
            )
            history.extend(reaction_history)
    elif "robot_rollout_training_horizon" in cfg:
        history = train_blockwise_horizons(
            candidate, batch, epochs=shared_epochs,
            learning_rate=float(cfg["learning_rate"]),
            robot_horizon=int(cfg["robot_rollout_training_horizon"]),
            object_horizon=int(cfg["object_rollout_training_horizon"]),
        )
    else:
        history = train_model(candidate, batch, component="shared",
                              epochs=shared_epochs, learning_rate=float(cfg["learning_rate"]),
                              rollout_horizon=int(cfg["rollout_training_horizon"]))
    if refinement_epochs:
        for name, parameter in candidate.named_parameters():
            if name.startswith("object_"):
                parameter.requires_grad_(False)
        print(f"[refine] joint-only epochs={refinement_epochs}", flush=True)
        refinement = train_model(
            candidate, batch, component="joint", epochs=refinement_epochs,
            learning_rate=float(cfg["joint_refinement_learning_rate"]),
            rollout_horizon=int(cfg.get("rollout_training_horizon", 5)),
        )
        history.extend(refinement)
    domain = next(x for x in protocol.test if x.domain_id == cfg["primary_domain"])
    index = list(protocol.test).index(domain)
    test_seed = args.seed * 100_000 + index * 1000 + 500
    test_key = json.dumps({"kind": "push_test", "seed": test_seed,
        "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
    test_data = cached_collect(args.cache_dir, test_key, lambda: collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=test_seed, targets=tuple(x.as_array() for x in targets.evaluation), **common))
    topology_methods = ("bt_dpwm",) if use_topology else ()
    rows = evaluate({"shared_baseline": baseline, "bt_dpwm": candidate}, domain, test_data,
                    device, int(q0a["rollout_horizon"]), topology_methods)
    result = {row["method"]: row for row in rows}; base, cand = result["shared_baseline"], result["bt_dpwm"]
    improvement = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
    obj, free, overall = improvement("object_rmse"), improvement("free_rmse"), improvement("overall_rmse")
    gate = cfg["gate"]
    passed = (obj >= gate["minimum_object_improvement_pct"]
              and free >= -gate["maximum_free_arm_regression_pct"]
              and overall >= gate["minimum_overall_improvement_pct"]
              and cand["violation_rmse"] <= gate["maximum_constraint_violation_rms"])
    summary = {"config_version": cfg["version"], "seed": args.seed, "device": str(device),
               "parameters": sum(p.numel() for p in candidate.parameters()),
               "shared_epochs": shared_epochs, "joint_refinement_epochs": refinement_epochs,
               "block_coordinate_training": bool(cfg.get("block_coordinate_training", False)),
               "reaction_epochs": int(cfg.get("reaction_epochs", 0)),
               "baseline_history": baseline_history,
               "swa_scaffold_history": swa_history, "swa_scaffold_count": swa_count,
               "scaffold_selection": scaffold_selection,
               "robot_selection": robot_selection,
               "object_selection": object_selection,
               "robot_blend_selection": blend_selection,
               "object_improvement_pct": obj, "free_arm_improvement_pct": free,
               "overall_improvement_pct": overall, "gate_passed": passed,
               "rows": rows, "history": history}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(candidate.state_dict(), args.output_dir / "model.pt")
    if bool(cfg.get("train_baseline", False)):
        torch.save(baseline.state_dict(), args.output_dir / "baseline_model.pt")
    print(f"[Y0] object={obj:+.2f}% free={free:+.2f}% overall={overall:+.2f}% "
          f"decision={'PASS' if passed else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
