"""Z0: frozen cross-domain and deployment-cost audit for BT-DPWM Y6."""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def _load_models(seed, device, y6_cfg, v0_dir, replication_root):
    cfg = TopologyGraphConfig(hidden_dim=int(y6_cfg["hidden_dim"]))
    baseline = TopologyGraphWorldModel(cfg).to(device)
    candidate = BlockTriangularDPWM(
        cfg, contact_conditioned_robot=True, independent_object_encoder=True).to(device)
    if seed == 7:
        baseline.load_state_dict(torch.load(v0_dir / "models.pt", map_location=device)["shared_compute_matched"])
        candidate_path = Path("runs/g2_bt_dpwm_gate_y6/seed7_v1/model.pt")
    else:
        run_dir = replication_root / f"seed{seed}_v1"
        baseline.load_state_dict(torch.load(run_dir / "baseline_model.pt", map_location=device))
        candidate_path = run_dir / "model.pt"
    candidate.load_state_dict(torch.load(candidate_path, map_location=device))
    return baseline.eval(), candidate.eval()


@torch.inference_mode()
def benchmark(model, device, batch_size, horizon, warmup, measured):
    state = torch.randn(batch_size, 14, device=device)
    action = torch.randn(batch_size, horizon, 5, device=device)
    mask = torch.zeros(batch_size, 5, device=device)
    angle = torch.zeros_like(mask)

    def rollout():
        prediction, hidden = state, None
        for depth in range(horizon):
            prediction, hidden = model.step(prediction, action[:, depth], mask, angle, hidden)

    for _ in range(warmup):
        rollout()
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measured): rollout()
        end.record(); torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        start_time = time.perf_counter()
        for _ in range(measured): rollout()
        elapsed_ms = 1000.0 * (time.perf_counter() - start_time)
        peak_bytes = None
    return {"batch_size": batch_size, "horizon": horizon,
            "rollout_ms": elapsed_ms / measured,
            "per_sample_step_us": elapsed_ms * 1000.0 / (measured * batch_size * horizon),
            "peak_memory_bytes": peak_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v0-run-dir", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    y6 = yaml.safe_load(Path(audit["y6_config"]).read_text(encoding="utf-8"))
    v0 = yaml.safe_load(Path(y6["v0_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    rows = []
    for seed in audit["seeds"]:
        baseline, candidate = _load_models(seed, device, y6, args.v0_run_dir, args.replication_root)
        for domain_id in audit["domains"]:
            domain = next(x for x in protocol.test if x.domain_id == domain_id)
            index = list(protocol.test).index(domain)
            test_seed = seed * 100_000 + index * 1000 + 500
            key = json.dumps({"kind": "push_test", "seed": test_seed,
                              "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
            trajectories = cached_collect(args.cache_dir, key, lambda d=domain, s=test_seed:
                collect_push_domains((d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                                     seed=s, targets=tuple(x.as_array() for x in targets.evaluation), **common))
            result = evaluate({"shared": baseline, "bt_dpwm": candidate}, domain,
                              trajectories, device, int(q0a["rollout_horizon"]))
            values = {x["method"]: x for x in result}
            base, cand = values["shared"], values["bt_dpwm"]
            improve = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
            rows.append({"seed": seed, "domain": domain_id,
                         "free_improvement_pct": improve("free_rmse"),
                         "object_improvement_pct": improve("object_rmse"),
                         "overall_improvement_pct": improve("overall_rmse"),
                         "constraint_violation_rms": cand["violation_rmse"],
                         "baseline": base, "candidate": cand})
    h96 = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=96)).to(device).eval()
    h136 = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=136)).to(device).eval()
    bt = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=96),
                            contact_conditioned_robot=True,
                            independent_object_encoder=True).to(device).eval()
    timing_cfg = audit["timing"]
    models = {"shared_h96": h96, "shared_h136_parameter_matched": h136, "bt_dpwm_h96": bt}
    costs = {name: {"parameters": sum(p.numel() for p in model.parameters()),
                    "timing": benchmark(model, device, int(timing_cfg["batch_size"]),
                                        int(timing_cfg["horizon"]),
                                        int(timing_cfg["warmup_rollouts"]),
                                        int(timing_cfg["measured_rollouts"]))}
             for name, model in models.items()}
    overall = [row["overall_improvement_pct"] for row in rows]
    gate_cfg = audit["cross_domain_gate"]
    gate = (float(np.mean(overall)) >= gate_cfg["minimum_mean_overall_improvement_pct"]
            and sum(x < 0 for x in overall) <= gate_cfg["maximum_domains_with_overall_regression"]
            and max(row["constraint_violation_rms"] for row in rows)
            <= gate_cfg["maximum_constraint_violation_rms"])
    summary = {"config_version": audit["version"], "device": str(device), "rows": rows,
               "mean_free_improvement_pct": float(np.mean([x["free_improvement_pct"] for x in rows])),
               "mean_object_improvement_pct": float(np.mean([x["object_improvement_pct"] for x in rows])),
               "mean_overall_improvement_pct": float(np.mean(overall)),
               "overall_regression_count": sum(x < 0 for x in overall),
               "cross_domain_gate_passed": gate, "costs": costs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Z0] mean free={summary['mean_free_improvement_pct']:+.2f}% "
          f"object={summary['mean_object_improvement_pct']:+.2f}% "
          f"overall={summary['mean_overall_improvement_pct']:+.2f}% "
          f"regressions={summary['overall_regression_count']}/12 "
          f"decision={'PASS' if gate else 'NO-GO'}", flush=True)
    for name, cost in costs.items():
        print(f"  {name}: params={cost['parameters']:,} "
              f"rollout={cost['timing']['rollout_ms']:.3f}ms", flush=True)


if __name__ == "__main__": main()
