"""V0: architecture/compute-matched shared-vs-independent graph transitions."""
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
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def _component_loss(prediction, target, component):
    joint = (prediction[:, :10] - target[:, :10]).pow(2).mean()
    obj = (prediction[:, 10:] - target[:, 10:]).pow(2).mean()
    if component == "joint":
        return joint
    if component == "object":
        return obj
    if component == "shared":
        return joint + obj
    raise ValueError(component)


def train_model(model, batch, *, component, epochs, learning_rate, rollout_horizon,
                use_topology=False):
    states, actions, mask, angle = batch
    zeros = torch.zeros_like(mask)
    model_mask, model_angle = (mask, angle) if use_topology else (zeros, zeros)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        hidden, one_step = None, []
        for step in range(actions.shape[1]):
            prediction, hidden = model.step(
                states[:, step], actions[:, step], model_mask, model_angle, hidden
            )
            one_step.append(_component_loss(prediction, states[:, step + 1], component))
        rollout = []
        horizon = min(rollout_horizon, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction, hidden = states[:, start], None
            for offset in range(horizon):
                prediction, hidden = model.step(
                    prediction, actions[:, start + offset], model_mask, model_angle, hidden
                )
                rollout.append(_component_loss(
                    prediction, states[:, start + offset + 1], component
                ))
        loss = torch.stack(one_step).mean() + 0.5 * torch.stack(rollout).mean()
        optimizer.zero_grad(); loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        history.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch={epoch + 1:03d} loss={float(loss.detach()):.6f} grad={gradient:.3f}",
                flush=True,
            )
    return history


@torch.no_grad()
def evaluate(models, domain, trajectories, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros = torch.zeros_like(mask)
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    methods = ("shared_parameter_matched", "shared_compute_matched", "independent_experts")
    horizon = min(horizon, actions.shape[1])
    values = {method: {depth: {"free": [], "object": [], "violation": []}
                       for depth in range(horizon)} for method in methods}
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction = {method: states[:, start].clone() for method in methods}
        hidden = {
            "shared_parameter_matched": None, "shared_compute_matched": None,
            "independent_joint": None, "independent_object": None,
        }
        for depth in range(horizon):
            action = actions[:, start + depth]
            target = states[:, start + depth + 1]
            for method in ("shared_parameter_matched", "shared_compute_matched"):
                raw, hidden[method] = models[method].step(
                    prediction[method], action, zeros, zeros, hidden[method]
                )
                prediction[method] = surgery.project_state(raw, mask, angle)
            joint, hidden["independent_joint"] = models["independent_joint"].step(
                prediction["independent_experts"], action, zeros, zeros,
                hidden["independent_joint"],
            )
            obj, hidden["independent_object"] = models["independent_object"].step(
                prediction["independent_experts"], action, zeros, zeros,
                hidden["independent_object"],
            )
            product = obj.clone(); product[:, :10] = joint[:, :10]
            prediction["independent_experts"] = surgery.project_state(product, mask, angle)
            for method in methods:
                error = (prediction[method] - target).pow(2)
                values[method][depth]["free"].append(
                    (error[:, :10] * free_mask).sum(dim=-1) / free_count
                )
                values[method][depth]["object"].append(error[:, 10:].mean(dim=-1))
                values[method][depth]["violation"].append(
                    surgery.constraint_violation(prediction[method], mask, angle).pow(2)
                )
    rows = []
    for method in methods:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen V0 list")
    q0a = yaml.safe_load(Path(config["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    common = dict(
        steps=int(q0a["steps"]), excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=calibration, **common,
    )
    batch = _batch(train, device)
    torch.manual_seed(args.seed)
    models = {
        "shared_parameter_matched": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(config["shared_parameter_matched_hidden_dim"])
        )).to(device),
        "shared_compute_matched": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(config["shared_compute_matched_hidden_dim"])
        )).to(device),
        "independent_joint": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(config["independent_hidden_dim"])
        )).to(device),
        "independent_object": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(config["independent_hidden_dim"])
        )).to(device),
    }
    specifications = (
        ("shared_parameter_matched", "shared", int(config["shared_parameter_matched_epochs"])),
        ("shared_compute_matched", "shared", int(config["shared_compute_matched_epochs"])),
        ("independent_joint", "joint", int(config["independent_epochs_per_expert"])),
        ("independent_object", "object", int(config["independent_epochs_per_expert"])),
    )
    histories = {}
    for name, component, epochs in specifications:
        print(f"[train] {name} component={component}", flush=True)
        histories[name] = train_model(
            models[name], batch, component=component, epochs=epochs,
            learning_rate=float(config["learning_rate"]),
            rollout_horizon=int(config["rollout_training_horizon"]),
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: model.state_dict() for name, model in models.items()}, args.output_dir / "models.pt")
    domain = next(item for item in protocol.test if item.domain_id == config["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 100_000 + index * 1000 + 500, targets=evaluation, **common,
    )
    rows = evaluate(models, domain, trajectories, device, int(q0a["rollout_horizon"]))
    final = {row["method"]: row for row in rows if row["depth"] == int(q0a["rollout_horizon"])}
    independent = final["independent_experts"]
    pct = lambda candidate, baseline: 100.0 * (baseline - candidate) / baseline
    vs_parameter = pct(independent["free_arm_rmse"], final["shared_parameter_matched"]["free_arm_rmse"])
    vs_compute = pct(independent["free_arm_rmse"], final["shared_compute_matched"]["free_arm_rmse"])
    best_shared_object = min(
        final["shared_parameter_matched"]["object_rmse"],
        final["shared_compute_matched"]["object_rmse"],
    )
    object_regression = 100.0 * (independent["object_rmse"] - best_shared_object) / best_shared_object
    gate = config["gate"]
    passed = (
        vs_parameter >= float(gate["minimum_free_improvement_vs_shared_parameter_matched_pct"])
        and vs_compute >= float(gate["minimum_free_improvement_vs_shared_compute_matched_pct"])
        and object_regression <= float(gate["maximum_object_regression_vs_best_shared_pct"])
        and independent["constraint_violation_rms"] <= float(gate["maximum_constraint_violation_rms"])
    )
    parameters = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    summary = {
        "config_version": config["version"], "seed": args.seed,
        "independent_free_improvement_vs_shared_parameter_matched_pct": vs_parameter,
        "independent_free_improvement_vs_shared_compute_matched_pct": vs_compute,
        "independent_object_regression_vs_best_shared_pct": object_regression,
        "gate_passed": passed, "parameters": parameters,
        "independent_total_parameters": parameters["independent_joint"] + parameters["independent_object"],
        "histories": histories, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(
        f"[V0] vs_param={vs_parameter:+.2f}% vs_compute={vs_compute:+.2f}% "
        f"object_reg={object_regression:+.2f}% decision={'PASS' if passed else 'NO-GO'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
