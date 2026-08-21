"""Z2: contact/free rollout and robot-hidden contact-information diagnosis."""
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
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import collect_push_domains


def load_models(device):
    shared = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=136)).to(device)
    shared.load_state_dict(torch.load(
        "runs/g2_bt_dpwm_param_match_z1/seed7_v1/shared_model.pt", map_location=device))
    models = {"shared_h136_240": shared}
    for name, path in (
        ("bt_y6", "runs/g2_bt_dpwm_gate_y6/seed7_v1/model.pt"),
        ("bt_y7", "runs/g2_bt_dpwm_robot_budget_y7/seed7_v1/model.pt"),
    ):
        model = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=96),
                                    contact_conditioned_robot=True,
                                    independent_object_encoder=True).to(device)
        model.load_state_dict(torch.load(path, map_location=device)); models[name] = model
    return {name: model.eval() for name, model in models.items()}


@torch.no_grad()
def rollout_contact_metrics(models, domain, trajectories, device, horizon):
    states = torch.stack([x.states for x in trajectories]).to(device)
    actions = torch.stack([x.actions for x in trajectories]).to(device)
    contacts = torch.stack([x.contact_mask for x in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros, surgery = torch.zeros_like(mask), TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    values = {name: {kind: {key: [] for key in ("free", "object")}
                     for kind in ("contact", "no_contact")} for name in models}
    depth_values = {name: {d: [] for d in range(horizon)} for name in models}
    oracle_depth = {name: {d: [] for d in range(horizon)}
                    for name in models if name.startswith("bt_")}
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction = {name: states[:, start].clone() for name in models}
        hidden = {name: None for name in models}
        oracle_prediction = {name: states[:, start].clone() for name in oracle_depth}
        oracle_hidden = {name: None for name in oracle_depth}
        for depth in range(horizon):
            target = states[:, start + depth + 1]
            label = contacts[:, start + depth]
            for name, model in models.items():
                raw, hidden[name] = model.step(prediction[name], actions[:, start + depth],
                                               zeros, zeros, hidden[name])
                prediction[name] = surgery.project_state(raw, mask, angle)
                error = (prediction[name] - target).pow(2)
                free = (error[:, :10] * free_mask).sum(-1) / free_count
                obj = error[:, 10:].mean(-1)
                depth_values[name][depth].append(free)
                for kind, select in (("contact", label), ("no_contact", ~label)):
                    if select.any():
                        values[name][kind]["free"].append(free[select])
                        values[name][kind]["object"].append(obj[select])
            for name in oracle_depth:
                raw, oracle_hidden[name] = models[name].step(
                    oracle_prediction[name], actions[:, start + depth],
                    zeros, zeros, oracle_hidden[name])
                oracle_free = ((raw[:, :10] - target[:, :10]).pow(2) * free_mask).sum(-1) / free_count
                oracle_depth[name][depth].append(oracle_free)
                # Isolate robot recursion by supplying the true next object state.
                oracle_prediction[name] = torch.cat((raw[:, :10], target[:, 10:]), -1)
    rows = []
    for name in models:
        row = {"method": name}
        for kind in ("contact", "no_contact"):
            for key in ("free", "object"):
                items = values[name][kind][key]
                row[f"{kind}_{key}_rmse"] = float(torch.cat(items).mean().sqrt())
            row[f"{kind}_count"] = sum(x.numel() for x in values[name][kind]["free"])
        row["free_rmse_by_depth"] = [float(torch.cat(depth_values[name][d]).mean().sqrt())
                                     for d in range(horizon)]
        rows.append(row)
        if name in oracle_depth:
            rows.append({"method": name + "_true_object_oracle",
                         "free_rmse_by_depth": [
                             float(torch.cat(oracle_depth[name][d]).mean().sqrt())
                             for d in range(horizon)]})
    return rows


@torch.no_grad()
def hidden_dataset(model, trajectories, device):
    features, labels, splits = [], [], []
    for index, trajectory in enumerate(trajectories):
        state = trajectory.states.to(device); action = trajectory.actions.to(device)
        mask = torch.zeros(1, 5, device=device); angle = torch.zeros_like(mask); hidden = None
        for step in range(len(action)):
            _, hidden = model.step(state[step:step+1], action[step:step+1], mask, angle, hidden)
            robot_hidden = hidden[:, :5] if hidden.shape[1] == 10 else hidden
            features.append(robot_hidden.mean(1).squeeze(0).cpu())
            labels.append(float(trajectory.contact_mask[step])); splits.append(index % 2)
    return torch.stack(features), torch.tensor(labels), torch.tensor(splits, dtype=torch.bool)


def auc(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]
    return float(((positive[:, None] > negative[None, :]).float()
                  + 0.5 * (positive[:, None] == negative[None, :]).float()).mean())


def linear_probe(features, labels, test_mask, steps, learning_rate):
    mean, std = features[~test_mask].mean(0), features[~test_mask].std(0).clamp_min(1e-6)
    x = (features - mean) / std
    probe = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    train_x, train_y = x[~test_mask], labels[~test_mask]
    pos_weight = (train_y == 0).sum() / (train_y == 1).sum().clamp_min(1)
    for _ in range(steps):
        logits = probe(train_x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, train_y, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        scores = probe(x[test_mask]).squeeze(-1); y = labels[test_mask]
        pred = scores >= 0
        tpr = (pred[y == 1]).float().mean(); tnr = (~pred[y == 0]).float().mean()
    return {"auc": auc(scores, y), "balanced_accuracy": float(0.5 * (tpr + tnr)),
            "train_count": int((~test_mask).sum()), "test_count": int(test_mask.sum()),
            "test_contact_count": int(y.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"])); targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); models = load_models(device)
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    train_key = json.dumps({"kind": "push_train", "seed": cfg["seed"],
        "domains": [x.domain_id for x in protocol.train], "q0a": q0a}, sort_keys=True)
    train = cached_collect(args.cache_dir, train_key, lambda: collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=cfg["seed"] * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common))
    probes = {}
    for name, model in models.items():
        x, y, split = hidden_dataset(model, train, device)
        probes[name] = linear_probe(x, y, split, int(cfg["probe_steps"]),
                                    float(cfg["probe_learning_rate"]))
    domain_rows = []
    for domain_id in cfg["domains"]:
        domain = next(x for x in protocol.test if x.domain_id == domain_id)
        index = list(protocol.test).index(domain); seed = cfg["seed"] * 100_000 + index * 1000 + 500
        key = json.dumps({"kind": "push_test", "seed": seed, "domain": domain_id, "q0a": q0a}, sort_keys=True)
        trajectories = cached_collect(args.cache_dir, key, lambda d=domain, s=seed:
            collect_push_domains((d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                                 seed=s, targets=tuple(x.as_array() for x in targets.evaluation), **common))
        for row in rollout_contact_metrics(models, domain, trajectories, device,
                                           int(cfg["rollout_horizon"])):
            domain_rows.append({"domain": domain_id, **row})
    summary = {"config_version": cfg["version"], "seed": cfg["seed"],
               "hidden_contact_probes": probes, "rows": domain_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[Z2 probes] " + " ".join(f"{k}:auc={v['auc']:.3f},bacc={v['balanced_accuracy']:.3f}"
          for k, v in probes.items()), flush=True)
    for name in models:
        selected = [r for r in domain_rows if r["method"] == name]
        print(f"  {name}: contact_free={np.mean([r['contact_free_rmse'] for r in selected]):.4f} "
              f"no_contact_free={np.mean([r['no_contact_free_rmse'] for r in selected]):.4f} "
              f"depth10={np.mean([r['free_rmse_by_depth'][-1] for r in selected]):.4f}", flush=True)
        if name.startswith("bt_"):
            oracle = [r for r in domain_rows if r["method"] == name + "_true_object_oracle"]
            print(f"    true-object oracle depth10="
                  f"{np.mean([r['free_rmse_by_depth'][-1] for r in oracle]):.4f}", flush=True)


if __name__ == "__main__": main()
