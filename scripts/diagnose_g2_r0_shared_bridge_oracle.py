"""Diagnose whether strict-to-shared latent mismatch causes R0 object drift.

This is an upper-bound diagnostic, not a deployable method: the hybrid keeps the
strict damage-projected robot transition but supplies its frozen object head with
the hidden bridge emitted by the frozen shared model on the same rollout state.
No target/test state is used after rollout initialization.
"""
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
from scripts.evaluate_g2_r0_core_metrics import evaluate
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import collect_push_domains


class SharedBridgeOracle(torch.nn.Module):
    """Run a frozen shared encoder only to replace the object bridge code."""

    def __init__(self, strict, shared):
        super().__init__()
        self.strict = strict
        self.shared = shared

    def step(self, state, action, mask, lock_angle, hidden):
        strict_hidden, shared_hidden = (None, None) if hidden is None else hidden
        robot, next_strict_hidden, obj, projected_action, depth = self.strict.step_robot(
            state, action, mask, lock_angle, strict_hidden)
        zeros = torch.zeros_like(mask)
        _, next_shared_hidden = self.shared.step(
            state, action, zeros, zeros, shared_hidden)
        prediction, _ = self.strict.step_object(
            robot, obj, projected_action, mask, lock_angle, depth,
            next_shared_hidden, previous_robot=state[:, :10])
        return prediction, (next_strict_hidden, next_shared_hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 25, 50])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(cfg["baseline_hidden_dim"]))).to(device)
    shared.load_state_dict(torch.load(str(
        cfg["external_baseline_model_template"]).format(seed=args.seed), map_location=device))
    strict = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=bool(cfg.get("contact_conditioned_robot", False)),
        independent_object_encoder=bool(cfg.get("independent_object_encoder", False)),
        compact_bridge_object_head=bool(cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(cfg.get("geometric_object_rank", 0)),
        intervention_residual_support_joints=tuple(
            int(x) for x in cfg.get("intervention_residual_support_joints", [])),
        intervention_residual_meta_train=bool(
            cfg.get("intervention_residual_meta_train", False)),
        intervention_object_rank=int(cfg.get("intervention_object_rank", 0)),
        object_bridge_alignment_rank=int(
            cfg.get("object_bridge_alignment_rank", 0))).to(device)
    strict.load_state_dict(torch.load(args.model, map_location=device))
    models = {"shared_projected": shared.eval(), "strict_bt": strict.eval(),
              "shared_bridge_oracle": SharedBridgeOracle(strict, shared).eval()}
    common = dict(steps=int(q0a["steps"]), excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        goal_exploration_std=float(q0a["goal_exploration_std"]))
    rows = []
    for index, domain in enumerate(protocol.test):
        seed = args.seed * 100_000 + index * 1000 + 500
        key = json.dumps({"kind": "push_test", "seed": seed,
                          "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
        trajectories = cached_collect(args.cache_dir, key, lambda domain=domain: collect_push_domains(
            (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            seed=seed, targets=tuple(x.as_array() for x in targets.evaluation), **common))
        rows.extend(evaluate(models, trajectories, domain, args.horizons, device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"version": "g2_r0_shared_bridge_oracle_v1",
        "seed": args.seed, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
