"""Select contact-selective geometry and integration blend on validation only."""
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
from scripts.run_bt_dpwm_gate_y0 import (aggregate_topology_losses, cached_collect,
    object_losses_per_trajectory, topology_group_indices)
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_push_benchmark import collect_push_domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    key = json.dumps({"kind": "push_validation", "seed": args.seed,
                      "domains": [x.domain_id for x in protocol.validation],
                      "q0a": q0a}, sort_keys=True)
    trajectories = cached_collect(args.cache_dir, key, lambda: collect_push_domains(
        protocol.validation, trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 10_000 + 750,
        targets=tuple(x.as_array() for x in targets.validation), **common))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=False, independent_object_encoder=True,
        object_hidden_dim=int(cfg["object_hidden_dim"]),
        geometric_object_rank=int(cfg["geometric_object_rank"]),
        object_integration_dt=0.005).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    batch, groups = _batch(trajectories, device), topology_group_indices(trajectories, device)
    candidates = []
    with torch.no_grad():
        for gated in (False, True):
            model.geometric_object_contact_gate = gated
            for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
                model.object_position_blend = blend
                horizon_losses, details = [], {}
                for horizon in (10, 25, 50):
                    per = object_losses_per_trajectory(model, batch, horizon, True)
                    value, topology = aggregate_topology_losses(per, groups, 0.5)
                    horizon_losses.append(value)
                    details[str(horizon)] = {"loss": float(value),
                        "topology_losses": {name: float(item) for name, item in topology.items()}}
                candidates.append({"contact_gate": gated, "blend": blend,
                    "mean_validation_loss": float(torch.stack(horizon_losses).mean()),
                    "per_horizon": details})
    selected = min(candidates, key=lambda item: item["mean_validation_loss"])
    result = {"version": "g2_r0_object_geometry_policy_selection_v1",
              "selection_split": "validation", "seed": args.seed,
              "candidates": candidates, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
