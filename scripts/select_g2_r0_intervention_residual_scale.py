"""Select one post-training intervention-residual scale on frozen validation data."""
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
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_bt_dpwm_gate_y0 import (
    _batch, aggregate_topology_losses, cached_collect,
    object_losses_per_trajectory, topology_group_indices,
)
from scripts.run_push_benchmark import collect_push_domains


def build_model(cfg, device):
    return BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=bool(cfg.get("contact_conditioned_robot", False)),
        contact_gated_object_context=bool(
            cfg.get("contact_gated_object_context", False)),
        independent_object_encoder=bool(cfg.get("independent_object_encoder", False)),
        object_hidden_dim=int(cfg.get("object_hidden_dim", cfg["hidden_dim"])),
        compact_bridge_object_head=bool(cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(cfg.get("geometric_object_rank", 0)),
        intervention_residual_support_joints=tuple(
            int(x) for x in cfg.get("intervention_residual_support_joints", [])),
        intervention_residual_meta_train=bool(
            cfg.get("intervention_residual_meta_train", False)),
        intervention_object_rank=int(cfg.get("intervention_object_rank", 0)),
        object_bridge_alignment_rank=int(cfg.get("object_bridge_alignment_rank", 0)),
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-template")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0, .125, .25, .375, .5, .625, .75, .875, 1])
    parser.add_argument("--horizons", nargs="+", type=int, default=[25, 50])
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.model_template is None) == (args.models is None):
        parser.error("provide exactly one of --model-template or --models")
    if args.models is not None and len(args.models) != len(args.seeds):
        parser.error("--models must have one path per seed")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for index, seed in enumerate(args.seeds):
        model = build_model(cfg, device)
        model_path = (args.model_template.format(seed=seed)
                      if args.model_template is not None else args.models[index])
        model.load_state_dict(torch.load(model_path, map_location=device))
        # Meta-training semantics intentionally activate the shared residual on
        # seen validation interventions; there are no stochastic train layers.
        model.train()
        common = dict(steps=int(q0a["steps"]), excitation="goal",
            block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
            goal_exploration_std=float(q0a["goal_exploration_std"]))
        key = json.dumps({"kind": "push_validation", "seed": seed,
            "domains": [x.domain_id for x in protocol.validation], "q0a": q0a},
            sort_keys=True)
        data = cached_collect(args.cache_dir, key, lambda: collect_push_domains(
            protocol.validation,
            trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            seed=seed * 10_000 + 750,
            targets=tuple(x.as_array() for x in targets.validation), **common))
        batch, groups = _batch(data, device), topology_group_indices(data, device)
        for scale in args.scales:
            model.intervention_residual_scale = float(scale)
            for horizon in args.horizons:
                with torch.no_grad():
                    value, topology = aggregate_topology_losses(
                        object_losses_per_trajectory(model, batch, horizon, True),
                        groups, 0.5)
                rows.append({"seed": seed, "scale": float(scale),
                    "horizon": int(horizon), "loss": float(value),
                    "topology_losses": {k: float(v) for k, v in topology.items()}})
    base = {(row["seed"], row["horizon"]): row["loss"] for row in rows
            if row["scale"] == 0.0}
    candidates = []
    for scale in args.scales:
        selected = [row for row in rows if row["scale"] == float(scale)]
        ratios = [row["loss"] / base[(row["seed"], row["horizon"])]
                  for row in selected]
        score = 0.5 * float(np.mean(ratios)) + 0.5 * float(np.max(ratios))
        candidates.append({"scale": float(scale), "score": score,
                           "mean_ratio": float(np.mean(ratios)),
                           "worst_ratio": float(np.max(ratios))})
    winner = min(candidates, key=lambda item: (item["score"], item["scale"]))
    output = {"version": "g2_r0_intervention_residual_scale_v1",
              "seeds": args.seeds, "horizons": args.horizons,
              "selected_scale": winner["scale"], "candidates": candidates,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[scale] selected={winner['scale']:.3f} score={winner['score']:.6f}")


if __name__ == "__main__":
    main()
