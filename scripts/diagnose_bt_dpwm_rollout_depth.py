"""Compare frozen BT-DPWM checkpoints by rollout depth without retraining."""
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
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_push_benchmark import collect_push_domains


@torch.no_grad()
def depth_rows(models, domain, trajectories, device, horizon):
    states = torch.stack([x.states for x in trajectories]).to(device)
    actions = torch.stack([x.actions for x in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros, surgery = torch.zeros_like(mask), TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), -1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    values = {name: {d: {k: [] for k in ("free", "object", "overall")}
             for d in range(horizon)} for name in models}
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred = {name: states[:, start].clone() for name in models}
        hidden = {name: None for name in models}
        for depth in range(horizon):
            target = states[:, start + depth + 1]
            for name, model in models.items():
                raw, hidden[name] = model.step(pred[name], actions[:, start + depth],
                                               zeros, zeros, hidden[name])
                pred[name] = surgery.project_state(raw, mask, angle)
                error = (pred[name] - target).pow(2)
                values[name][depth]["free"].append(
                    (error[:, :10] * free_mask).sum(-1) / free_count)
                values[name][depth]["object"].append(error[:, 10:].mean(-1))
                values[name][depth]["overall"].append(error.mean(-1))
    rows = []
    for name in models:
        for depth in range(horizon):
            rows.append({"method": name, "depth": depth + 1, **{
                f"{key}_rmse": float(torch.cat(items).mean().sqrt())
                for key, items in values[name][depth].items()}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--v0-run-dir", type=Path, required=True)
    parser.add_argument("--y2-run-dir", type=Path, required=True)
    parser.add_argument("--y3-run-dir", type=Path, required=True)
    parser.add_argument("--y4-run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    v0 = yaml.safe_load(Path(cfg["v0_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))
    baseline = TopologyGraphWorldModel(model_cfg).to(device)
    baseline.load_state_dict(torch.load(args.v0_run_dir / "models.pt", map_location=device)["shared_compute_matched"])
    models = {"shared": baseline}
    for name, run_dir in (("y2", args.y2_run_dir), ("y3", args.y3_run_dir)):
        model = BlockTriangularDPWM(model_cfg, contact_conditioned_robot=True,
                                    independent_object_encoder=True).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        models[name] = model
    if args.y4_run_dir is not None:
        y4 = BlockTriangularDPWM(model_cfg, contact_conditioned_robot=True,
                                independent_object_encoder=True).to(device)
        y4.load_state_dict(torch.load(args.y4_run_dir / "model.pt", map_location=device))
        models["y4"] = y4
        stitched = copy.deepcopy(y4)
        y2_state = models["y2"].state_dict()
        stitched_state = stitched.state_dict()
        for name in stitched_state:
            if name.startswith("object_"):
                stitched_state[name] = y2_state[name]
        stitched.load_state_dict(stitched_state)
        models["y4_robot_y2_object"] = stitched
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    domain = next(x for x in protocol.test if x.domain_id == cfg["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 100_000 + index * 1000 + 500,
        targets=tuple(x.as_array() for x in targets.evaluation), steps=int(q0a["steps"]),
        excitation="goal", block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        goal_exploration_std=float(q0a["goal_exploration_std"]))
    rows = depth_rows(models, domain, trajectories, device, int(q0a["rollout_horizon"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    for depth in (1, 5, 10):
        selected = [r for r in rows if r["depth"] == depth]
        print(f"[depth={depth}] " + " ".join(
            f"{r['method']}:free={r['free_rmse']:.4f},obj={r['object_rmse']:.4f}"
            for r in selected), flush=True)


if __name__ == "__main__":
    main()
