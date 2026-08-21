"""Z1: compare frozen BT-DPWM with a parameter- and epoch-matched shared graph."""
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
from scripts.run_dual_expert_fair_gate_v0 import train_model
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_push_benchmark import collect_push_domains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in cfg["seeds"]:
        raise ValueError("seed not in frozen Z1 list")
    y6 = yaml.safe_load(Path(cfg["y6_config"]).read_text(encoding="utf-8"))
    v0 = yaml.safe_load(Path(y6["v0_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    train_key = json.dumps({"kind": "push_train", "seed": args.seed,
        "domains": [x.domain_id for x in protocol.train], "q0a": q0a}, sort_keys=True)
    train_data = cached_collect(args.cache_dir, train_key, lambda: collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common))
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(cfg["shared_hidden_dim"]))).to(device)
    history = train_model(shared, _batch(train_data, device), component="shared",
                          epochs=int(cfg["shared_epochs"]),
                          learning_rate=float(cfg["shared_learning_rate"]),
                          rollout_horizon=int(cfg["shared_rollout_training_horizon"]))
    bt = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=int(y6["hidden_dim"])),
                            contact_conditioned_robot=True,
                            independent_object_encoder=True).to(device)
    bt_path = (Path("runs/g2_bt_dpwm_gate_y6/seed7_v1/model.pt") if args.seed == 7
               else args.replication_root / f"seed{args.seed}_v1" / "model.pt")
    bt.load_state_dict(torch.load(bt_path, map_location=device))
    rows = []
    for domain_id in cfg["domains"]:
        domain = next(x for x in protocol.test if x.domain_id == domain_id)
        index = list(protocol.test).index(domain)
        test_seed = args.seed * 100_000 + index * 1000 + 500
        key = json.dumps({"kind": "push_test", "seed": test_seed,
                          "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
        trajectories = cached_collect(args.cache_dir, key, lambda d=domain, s=test_seed:
            collect_push_domains((d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                                 seed=s, targets=tuple(x.as_array() for x in targets.evaluation), **common))
        result = evaluate({"shared_h136_240": shared, "bt_dpwm": bt}, domain,
                          trajectories, device, int(q0a["rollout_horizon"]))
        values = {x["method"]: x for x in result}; base, cand = values["shared_h136_240"], values["bt_dpwm"]
        improve = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
        rows.append({"domain": domain_id, "free_improvement_pct": improve("free_rmse"),
                     "object_improvement_pct": improve("object_rmse"),
                     "overall_improvement_pct": improve("overall_rmse"),
                     "constraint_violation_rms": cand["violation_rmse"],
                     "baseline": base, "candidate": cand})
    overall = [x["overall_improvement_pct"] for x in rows]; gate_cfg = cfg["gate"]
    passed = (float(np.mean(overall)) >= gate_cfg["minimum_mean_overall_improvement_pct"]
              and sum(x < 0 for x in overall) <= gate_cfg["maximum_domains_with_overall_regression"]
              and max(x["constraint_violation_rms"] for x in rows)
              <= gate_cfg["maximum_constraint_violation_rms"])
    summary = {"config_version": cfg["version"], "seed": args.seed, "device": str(device),
               "shared_parameters": sum(p.numel() for p in shared.parameters()),
               "bt_parameters": sum(p.numel() for p in bt.parameters()),
               "mean_free_improvement_pct": float(np.mean([x["free_improvement_pct"] for x in rows])),
               "mean_object_improvement_pct": float(np.mean([x["object_improvement_pct"] for x in rows])),
               "mean_overall_improvement_pct": float(np.mean(overall)),
               "overall_regression_count": sum(x < 0 for x in overall),
               "gate_passed": passed, "rows": rows, "history": history}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(shared.state_dict(), args.output_dir / "shared_model.pt")
    print(f"[Z1 seed={args.seed}] free={summary['mean_free_improvement_pct']:+.2f}% "
          f"object={summary['mean_object_improvement_pct']:+.2f}% "
          f"overall={summary['mean_overall_improvement_pct']:+.2f}% "
          f"regressions={summary['overall_regression_count']}/4 "
          f"decision={'PASS' if passed else 'NO-GO'}", flush=True)


if __name__ == "__main__": main()
