"""Evaluate strict-triangular G2-R0 on domain-specific multi-horizon metrics."""
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

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.contact_geometry import pusher_reference_point
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import collect_push_domains


@torch.no_grad()
def evaluate(models, trajectories, domain, horizons, device):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros, surgery = torch.zeros_like(mask), TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    rows = []
    for horizon in horizons:
        horizon = min(int(horizon), actions.shape[1])
        values = {name: {key: [] for key in
                  ("free", "object", "overall", "violation", "pusher_xy")}
                  for name in models}
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            predictions = {name: states[:, start].clone() for name in models}
            hidden = {name: None for name in models}
            for depth in range(horizon):
                target = states[:, start + depth + 1]
                for name, model in models.items():
                    aware = name != "shared_projected"
                    raw, hidden[name] = model.step(
                        predictions[name], actions[:, start + depth],
                        mask if aware else zeros, angle if aware else zeros,
                        hidden[name])
                    predictions[name] = surgery.project_state(raw, mask, angle)
                    if depth != horizon - 1:
                        continue
                    error = (predictions[name] - target).pow(2)
                    values[name]["free"].append(
                        (error[:, :10] * free_mask).sum(-1) / free_count)
                    values[name]["object"].append(error[:, 10:].mean(-1))
                    values[name]["overall"].append(error.mean(-1))
                    values[name]["violation"].append(
                        surgery.constraint_violation(predictions[name], mask, angle).pow(2))
                    pusher_error = (pusher_reference_point(predictions[name][:, :5])[..., :2]
                                     - pusher_reference_point(target[:, :5])[..., :2]).pow(2).mean(-1)
                    values[name]["pusher_xy"].append(pusher_error)
        for name, metrics in values.items():
            rows.append({"domain": domain.domain_id, "horizon": horizon, "method": name,
                         **{f"{key}_rmse": float(torch.cat(items).mean().sqrt())
                            for key, items in metrics.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 25, 50])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(cfg["baseline_hidden_dim"]))).to(device)
    shared.load_state_dict(torch.load(
        str(cfg["external_baseline_model_template"]).format(seed=args.seed),
        map_location=device))
    strict = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=False, independent_object_encoder=True,
        object_hidden_dim=int(cfg["object_hidden_dim"]),
        geometric_object_rank=int(cfg["geometric_object_rank"]),
        object_integration_dt=cfg.get("object_integration_dt"),
        object_position_blend=float(cfg.get("object_position_blend", 0.0))).to(device)
    strict.load_state_dict(torch.load(args.model, map_location=device))
    ablated = copy.deepcopy(strict)
    with torch.no_grad():
        for parameter in ablated.geometric_object_head.parameters():
            parameter.zero_()
    models = {"shared_projected": shared.eval(), "strict_bt": strict.eval(),
              "strict_bt_no_geometry": ablated.eval()}
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    rows = []
    for index, domain in enumerate(protocol.test):
        test_seed = args.seed * 100_000 + index * 1000 + 500
        key = json.dumps({"kind": "push_test", "seed": test_seed,
                          "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
        trajectories = cached_collect(args.cache_dir, key, lambda domain=domain: collect_push_domains(
            (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            seed=test_seed, targets=tuple(x.as_array() for x in targets.evaluation), **common))
        rows.extend(evaluate(models, trajectories, domain, args.horizons, device))
    output = {"version": "g2_r0_core_metrics_v1", "seed": args.seed,
              "horizons": args.horizons, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[R0] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
