"""Strict multi-domain audit for the frozen scaffold-specialize BT-DPWM."""
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
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_push_benchmark import collect_push_domains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reaction-gate-threshold", type=float)
    parser.add_argument("--reaction-gate-temperature", type=float)
    parser.add_argument("--reaction-scale", type=float)
    parser.add_argument("--reaction-event-decay", type=float)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(x) for x in args.seeds.split(",")]
    if any(seed not in cfg["seeds"] for seed in seeds): raise ValueError("seed outside frozen list")
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"])); targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    rows = []
    for seed in seeds:
        baseline = TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(cfg["robot_hidden_dim"]))).to(device)
        baseline.load_state_dict(torch.load(str(cfg["baseline_model_template"]).format(seed=seed), map_location=device))
        candidate = BlockTriangularDPWM(
            TopologyGraphConfig(hidden_dim=int(cfg["robot_hidden_dim"])),
            contact_conditioned_robot=True, independent_object_encoder=True,
            object_hidden_dim=int(cfg["object_hidden_dim"]),
            reaction_rank=int(cfg.get("reaction_rank", 0)),
            reaction_geometry_gate=bool(cfg.get("reaction_geometry_gate", False)),
            reaction_gate_threshold=(args.reaction_gate_threshold if args.reaction_gate_threshold is not None
                                     else float(cfg.get("reaction_gate_threshold", -0.005))),
            reaction_gate_temperature=(args.reaction_gate_temperature if args.reaction_gate_temperature is not None
                                       else float(cfg.get("reaction_gate_temperature", 0.002))),
            reaction_scale=(args.reaction_scale if args.reaction_scale is not None
                            else float(cfg.get("reaction_scale", 1.0))),
            reaction_physical_features=bool(cfg.get("reaction_physical_features", False)),
            reaction_event_decay=(args.reaction_event_decay if args.reaction_event_decay is not None
                                  else cfg.get("reaction_event_decay"))).to(device)
        candidate.load_state_dict(torch.load(str(cfg["candidate_model_template"]).format(seed=seed), map_location=device))
        baseline.eval(); candidate.eval()
        for domain_id in cfg["domains"]:
            domain = next(x for x in protocol.test if x.domain_id == domain_id)
            index = list(protocol.test).index(domain); test_seed = seed * 100_000 + index * 1000 + 500
            key = json.dumps({"kind": "push_test", "seed": test_seed, "domain": domain_id, "q0a": q0a}, sort_keys=True)
            trajectories = cached_collect(args.cache_dir, key, lambda d=domain, s=test_seed:
                collect_push_domains((d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                                     seed=s, targets=tuple(x.as_array() for x in targets.evaluation), **common))
            result = evaluate({"shared": baseline, "bt_dpwm": candidate}, domain,
                              trajectories, device, int(q0a["rollout_horizon"]))
            values = {x["method"]: x for x in result}; base, cand = values["shared"], values["bt_dpwm"]
            improve = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
            rows.append({"seed": seed, "domain": domain_id,
                         "free_improvement_pct": improve("free_rmse"),
                         "object_improvement_pct": improve("object_rmse"),
                         "overall_improvement_pct": improve("overall_rmse"),
                         "constraint_violation_rms": cand["violation_rmse"]})
    means = {key: float(np.mean([r[f"{key}_improvement_pct"] for r in rows]))
             for key in ("free", "object", "overall")}
    regressions = sum(r["overall_improvement_pct"] < 0 for r in rows); gate = cfg["gate"]
    passed = (means["free"] > gate["minimum_mean_free_improvement_pct"]
              and means["object"] > gate["minimum_mean_object_improvement_pct"]
              and means["overall"] >= gate["minimum_mean_overall_improvement_pct"]
              and regressions <= gate["maximum_overall_regression_count"]
              and max(r["constraint_violation_rms"] for r in rows) <= gate["maximum_constraint_violation_rms"])
    summary = {"config_version": cfg["version"], "seeds": seeds, "rows": rows,
               "reaction_gate_threshold": args.reaction_gate_threshold,
               "reaction_gate_temperature": args.reaction_gate_temperature,
               "reaction_scale": args.reaction_scale,
               "reaction_event_decay": args.reaction_event_decay,
               "mean_free_improvement_pct": means["free"],
               "mean_object_improvement_pct": means["object"],
               "mean_overall_improvement_pct": means["overall"],
               "overall_regression_count": regressions, "strict_gate_passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Z4] seeds={seeds} free={means['free']:+.2f}% object={means['object']:+.2f}% "
          f"overall={means['overall']:+.2f}% regressions={regressions}/{len(rows)} "
          f"decision={'PASS' if passed else 'NO-GO'}", flush=True)


if __name__ == "__main__": main()
