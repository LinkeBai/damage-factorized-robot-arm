"""Y0: fair single-model gate for the first executable BT-DPWM."""
from __future__ import annotations

import argparse
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
    train_data = collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common)
    model_cfg = TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))
    baseline = TopologyGraphWorldModel(model_cfg).to(device)
    baseline.load_state_dict(torch.load(args.v0_run_dir / "models.pt", map_location=device)["shared_compute_matched"])
    torch.manual_seed(args.seed)
    candidate = BlockTriangularDPWM(
        model_cfg,
        contact_conditioned_robot=bool(cfg.get("contact_conditioned_robot", False)),
        independent_object_encoder=bool(cfg.get("independent_object_encoder", False)),
    ).to(device)
    batch = _batch(train_data, device)
    refinement_epochs = int(cfg.get("joint_refinement_epochs", 0))
    shared_epochs = int(cfg["epochs"]) - refinement_epochs
    if bool(cfg.get("block_coordinate_training", False)):
        for name, parameter in candidate.named_parameters():
            if name.startswith("object_"):
                parameter.requires_grad_(False)
        print("[block 1/2] robot", flush=True)
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
    test_data = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 100_000 + index * 1000 + 500,
        targets=tuple(x.as_array() for x in targets.evaluation), **common)
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
               "object_improvement_pct": obj, "free_arm_improvement_pct": free,
               "overall_improvement_pct": overall, "gate_passed": passed,
               "rows": rows, "history": history}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(candidate.state_dict(), args.output_dir / "model.pt")
    print(f"[Y0] object={obj:+.2f}% free={free:+.2f}% overall={overall:+.2f}% "
          f"decision={'PASS' if passed else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
