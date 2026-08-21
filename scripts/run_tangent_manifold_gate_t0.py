"""Gate T0: explicit tangent dynamics versus projected joint-expert controls."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.tangent_manifold_graph import TangentManifoldGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _batch, _contexts, _load_ensemble
from scripts.run_ftgwm_gate_k1 import _train
from scripts.run_manifold_stabilization_gate_s0 import _ensemble_step
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


METHODS = ("projected_matched", "topology_projected", "tangent_manifold")


@torch.no_grad()
def evaluate(ensemble, models, domain, trajectories, ranges, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros = torch.zeros_like(mask)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    horizon = min(horizon, actions.shape[1])
    values = {method: {depth: {"free": [], "object": [], "violation": []}
                       for depth in range(horizon)} for method in METHODS}
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predictions = {method: states[:, start].clone() for method in METHODS}
        object_hidden = {method: [None] * len(ensemble) for method in METHODS}
        joint_hidden = {method: None for method in METHODS}
        for depth in range(horizon):
            action = actions[:, start + depth]
            target = states[:, start + depth + 1]
            for method in METHODS:
                state = predictions[method]
                object_mean, object_hidden[method] = _ensemble_step(
                    ensemble, state, action, contexts, object_hidden[method]
                )
                use_damage = method != "projected_matched"
                joint_mean, joint_hidden[method] = models[method].step(
                    state, action, mask if use_damage else zeros,
                    angle if use_damage else zeros, joint_hidden[method],
                )
                prediction = object_mean.clone()
                prediction[:, :10] = joint_mean[:, :10]
                prediction = surgery.project_state(prediction, mask, angle)
                predictions[method] = prediction
                error = (prediction - target).pow(2)
                values[method][depth]["free"].append(
                    (error[:, :10] * free_mask).sum(dim=-1) / free_count
                )
                values[method][depth]["object"].append(error[:, 10:].mean(dim=-1))
                values[method][depth]["violation"].append(
                    surgery.constraint_violation(prediction, mask, angle).pow(2)
                )
    rows = []
    for method in METHODS:
        for depth in range(horizon):
            rmse = lambda key: float(torch.cat(values[method][depth][key]).mean().sqrt())
            rows.append({
                "method": method, "depth": depth + 1,
                "free_arm_rmse": rmse("free"), "object_rmse": rmse("object"),
                "constraint_violation_rms": rmse("violation"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q0a-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--matched-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen T0 list")
    q0a = yaml.safe_load(Path(config["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    common = dict(
        steps=int(q0a["steps"]), excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=calibration, **common,
    )
    ensemble = _load_ensemble(args.q0a_checkpoint_dir / "ordinary_ensemble.pt", device)
    model_config = TopologyGraphConfig(hidden_dim=int(q0a["hidden_dim"]))
    torch.manual_seed(args.seed)
    topology = TopologyGraphWorldModel(model_config).to(device)
    tangent = TangentManifoldGraphWorldModel(model_config).to(device)
    tangent.load_state_dict(topology.state_dict())
    matched = TopologyGraphWorldModel(model_config).to(device)
    matched.load_state_dict(torch.load(
        args.matched_checkpoint, map_location=device, weights_only=True
    )["model_state_dict"])
    batch = _batch(train, device)
    for name, model in (("topology_projected", topology), ("tangent_manifold", tangent)):
        print(f"[train] {name}", flush=True)
        _train(
            model, batch, epochs=int(config["epochs"]),
            learning_rate=float(config["learning_rate"]), use_topology=True,
            include_object_loss=False, object_loss_weight=0.0,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": topology.state_dict()}, args.output_dir / "topology_projected.pt")
    torch.save({"model_state_dict": tangent.state_dict()}, args.output_dir / "tangent_manifold.pt")
    domain = next(item for item in protocol.test if item.domain_id == config["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 100_000 + index * 1000 + 500, targets=evaluation, **common,
    )
    rows = evaluate(
        ensemble,
        {"projected_matched": matched, "topology_projected": topology, "tangent_manifold": tangent},
        domain, trajectories, ranges, device, int(q0a["rollout_horizon"]),
    )
    final = {row["method"]: row for row in rows if row["depth"] == int(q0a["rollout_horizon"])}
    pct = lambda candidate, baseline: 100.0 * (baseline - candidate) / baseline
    tangent_final = final["tangent_manifold"]
    vs_matched = pct(tangent_final["free_arm_rmse"], final["projected_matched"]["free_arm_rmse"])
    vs_topology = pct(tangent_final["free_arm_rmse"], final["topology_projected"]["free_arm_rmse"])
    object_regression = -pct(tangent_final["object_rmse"], final["projected_matched"]["object_rmse"])
    gate = config["gate"]
    passed = (
        vs_matched >= float(gate["minimum_tangent_free_improvement_vs_projected_matched_pct"])
        and vs_topology >= float(gate["minimum_tangent_free_improvement_vs_topology_projected_pct"])
        and object_regression <= float(gate["maximum_object_regression_vs_projected_matched_pct"])
        and tangent_final["constraint_violation_rms"] <= float(gate["maximum_constraint_violation_rms"])
    )
    summary = {
        "config_version": config["version"], "seed": args.seed,
        "tangent_free_improvement_vs_projected_matched_pct": vs_matched,
        "tangent_free_improvement_vs_topology_projected_pct": vs_topology,
        "tangent_object_regression_vs_projected_matched_pct": object_regression,
        "gate_passed": passed, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(
        f"[T0] vs_matched={vs_matched:+.2f}% vs_topology={vs_topology:+.2f}% "
        f"object_reg={object_regression:+.2f}% decision={'PASS' if passed else 'NO-GO'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
