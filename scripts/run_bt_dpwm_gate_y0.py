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
from scripts.run_dual_expert_fair_gate_v0 import train_model
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


def robot_losses(model, batch, horizon):
    states, actions, mask, _ = batch
    zeros = torch.zeros_like(mask)
    one_step, hidden = [], None
    for step in range(actions.shape[1]):
        robot, hidden, _, _, _ = model.step_robot(
            states[:, step], actions[:, step], zeros, zeros, hidden)
        one_step.append((robot - states[:, step + 1, :10]).pow(2).mean())
    rollout = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden = states[:, start], None
        for offset in range(horizon):
            robot, hidden, obj, _, _ = model.step_robot(
                prediction, actions[:, start + offset], zeros, zeros, hidden)
            rollout.append((robot - states[:, start + offset + 1, :10]).pow(2).mean())
            prediction = torch.cat((robot, obj), -1)
    return torch.stack(one_step).mean() + 0.5 * torch.stack(rollout).mean()


def train_robot_only(model, batch, *, epochs, learning_rate, horizon):
    parameters = [p for name, p in model.named_parameters() if name.startswith("robot_")]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    for epoch in range(epochs):
        loss = robot_losses(model, batch, horizon)
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        optimizer.step(); history.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch+1:03d} loss={loss.item():.6f} grad={gradient:.3f}", flush=True)
    return history


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
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
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
    ).to(device)
    if bool(cfg.get("initialize_robot_from_baseline", False)):
        source = baseline.state_dict(); target = candidate.state_dict()
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
    batch = _batch(train_data, device)
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
            robot_history = train_robot_only(
                candidate, batch, epochs=int(cfg["robot_epochs"]),
                learning_rate=float(cfg["learning_rate"]),
                horizon=int(cfg["robot_rollout_training_horizon"]),
            )
        else:
            robot_history = train_model(
                candidate, batch, component="joint", epochs=int(cfg["robot_epochs"]),
                learning_rate=float(cfg["learning_rate"]),
                rollout_horizon=int(cfg["robot_rollout_training_horizon"]),
            )
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(name.startswith("object_"))
        print("[block 2/2] object on frozen robot rollouts", flush=True)
        object_history = train_model(
            candidate, batch, component="object", epochs=int(cfg["object_epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            rollout_horizon=int(cfg["object_rollout_training_horizon"]),
        )
        history = robot_history + object_history
        reaction_epochs = int(cfg.get("reaction_epochs", 0))
        if reaction_epochs:
            for name, parameter in candidate.named_parameters():
                parameter.requires_grad_(name.startswith("reaction_adapter."))
            print(f"[block 3/3] reaction adapter epochs={reaction_epochs}", flush=True)
            reaction_history = train_model(
                candidate, batch, component="joint", epochs=reaction_epochs,
                learning_rate=float(cfg["reaction_learning_rate"]),
                rollout_horizon=int(cfg["robot_rollout_training_horizon"]),
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
    rows = evaluate({"shared_baseline": baseline, "bt_dpwm": candidate}, domain, test_data,
                    device, int(q0a["rollout_horizon"]))
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
