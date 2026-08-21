"""X1: optimize the frozen shared graph with object-preserving gradient projection."""
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

from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_push_benchmark import collect_push_domains


def _losses(model, batch, horizon):
    states, actions, mask, _ = batch
    zeros = torch.zeros_like(mask)
    one_joint, one_object = [], []
    hidden = None
    for step in range(actions.shape[1]):
        pred, hidden = model.step(states[:, step], actions[:, step], zeros, zeros, hidden)
        error = (pred - states[:, step + 1]).pow(2)
        one_joint.append(error[:, :10].mean()); one_object.append(error[:, 10:].mean())
    roll_joint, roll_object = [], []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred, hidden = states[:, start], None
        for offset in range(horizon):
            pred, hidden = model.step(pred, actions[:, start + offset], zeros, zeros, hidden)
            error = (pred - states[:, start + offset + 1]).pow(2)
            roll_joint.append(error[:, :10].mean()); roll_object.append(error[:, 10:].mean())
    joint = torch.stack(one_joint).mean() + 0.5 * torch.stack(roll_joint).mean()
    obj = torch.stack(one_object).mean() + 0.5 * torch.stack(roll_object).mean()
    return joint, obj


def train(model, batch, *, epochs, learning_rate, horizon):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    parameters = list(model.parameters())
    history, conflict_epochs = [], 0
    for epoch in range(epochs):
        joint, obj = _losses(model, batch, horizon)
        joint_grad = torch.autograd.grad(joint, parameters, retain_graph=True, allow_unused=True)
        object_grad = torch.autograd.grad(obj, parameters, allow_unused=True)
        shared = [i for i, (name, _) in enumerate(model.named_parameters())
                  if "joint_head" not in name and "object_head" not in name]
        dot = sum((joint_grad[i] * object_grad[i]).sum() for i in shared
                  if joint_grad[i] is not None and object_grad[i] is not None)
        object_norm_sq = sum(object_grad[i].pow(2).sum() for i in shared
                             if object_grad[i] is not None)
        coefficient = torch.clamp(-dot / object_norm_sq.clamp_min(1e-12), min=0.0)
        if float(dot.detach()) < 0:
            conflict_epochs += 1
        optimizer.zero_grad(set_to_none=True)
        for i, parameter in enumerate(parameters):
            gj, go = joint_grad[i], object_grad[i]
            if gj is None and go is None:
                continue
            if i in shared and gj is not None and go is not None:
                parameter.grad = gj + coefficient * go + go
            elif gj is None:
                parameter.grad = go
            elif go is None:
                parameter.grad = gj
            else:
                parameter.grad = gj + go
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        optimizer.step()
        history.append({"joint": float(joint.detach()), "object": float(obj.detach()),
                        "dot": float(dot.detach()), "coefficient": float(coefficient.detach())})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch+1:03d} joint={joint.item():.6f} object={obj.item():.6f} "
                  f"dot={dot.item():+.3e} coef={coefficient.item():.3f} grad={gradient:.3f}", flush=True)
    return history, conflict_epochs


@torch.no_grad()
def evaluate(models, domain, trajectories, device, horizon, topology_aware_methods=()):
    states = torch.stack([x.states for x in trajectories]).to(device)
    actions = torch.stack([x.actions for x in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros, surgery = torch.zeros_like(mask), TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    values = {name: {k: [] for k in ("free", "object", "overall", "violation")} for name in models}
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred = {name: states[:, start].clone() for name in models}; hidden = {name: None for name in models}
        for depth in range(horizon):
            target = states[:, start + depth + 1]
            for name, model in models.items():
                model_mask, model_angle = ((mask, angle) if name in topology_aware_methods
                                           else (zeros, zeros))
                raw, hidden[name] = model.step(pred[name], actions[:, start + depth],
                                               model_mask, model_angle, hidden[name])
                pred[name] = surgery.project_state(raw, mask, angle)
                if depth == horizon - 1:
                    error = (pred[name] - target).pow(2)
                    values[name]["free"].append((error[:, :10] * free_mask).sum(-1) / free_count)
                    values[name]["object"].append(error[:, 10:].mean(-1))
                    values[name]["overall"].append(error.mean(-1))
                    values[name]["violation"].append(surgery.constraint_violation(pred[name], mask, angle).pow(2))
    return [{"method": name, **{f"{key}_rmse": float(torch.cat(items).mean().sqrt())
             for key, items in metrics.items()}} for name, metrics in values.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v0-run-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    v0 = yaml.safe_load(Path(cfg["v0_config"]).read_text(encoding="utf-8")); q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"])); targets = load_target_split(Path(q0a["targets"]))
    common = dict(steps=int(q0a["steps"]), excitation="goal", block_initial_xy=np.asarray(q0a["block_initial_xy"], float), goal_exploration_std=float(q0a["goal_exploration_std"]))
    train_data = collect_push_domains(protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]), seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common)
    baseline = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))).to(device)
    candidate = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))).to(device)
    baseline.load_state_dict(torch.load(args.v0_run_dir / "models.pt", map_location=device)["shared_compute_matched"])
    torch.manual_seed(args.seed); candidate = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))).to(device)
    history, conflicts = train(candidate, _batch(train_data, device), epochs=int(cfg["epochs"]), learning_rate=float(cfg["learning_rate"]), horizon=int(cfg["rollout_training_horizon"]))
    domain = next(x for x in protocol.test if x.domain_id == cfg["primary_domain"]); index = list(protocol.test).index(domain)
    test_data = collect_push_domains((domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]), seed=args.seed * 100_000 + index * 1000 + 500, targets=tuple(x.as_array() for x in targets.evaluation), **common)
    rows = evaluate({"shared_baseline": baseline, "object_preserving": candidate}, domain, test_data, device, int(q0a["rollout_horizon"]))
    result = {x["method"]: x for x in rows}; base, cand = result["shared_baseline"], result["object_preserving"]
    improve = lambda key: 100 * (base[key] - cand[key]) / base[key]
    obj_imp, free_imp, overall_imp = improve("object_rmse"), improve("free_rmse"), improve("overall_rmse")
    gate = cfg["gate"]; passed = obj_imp >= gate["minimum_object_improvement_pct"] and free_imp >= -gate["maximum_free_arm_regression_pct"] and overall_imp >= gate["minimum_overall_improvement_pct"] and cand["violation_rmse"] <= gate["maximum_constraint_violation_rms"]
    summary = {"config_version": cfg["version"], "seed": args.seed, "device": str(device), "conflict_epochs": conflicts, "object_improvement_pct": obj_imp, "free_arm_improvement_pct": free_imp, "overall_improvement_pct": overall_imp, "gate_passed": passed, "rows": rows, "history": history}
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(candidate.state_dict(), args.output_dir / "model.pt")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"[X1] object={obj_imp:+.2f}% free={free_imp:+.2f}% overall={overall_imp:+.2f}% decision={'PASS' if passed else 'NO-GO'}", flush=True)


if __name__ == "__main__": main()
