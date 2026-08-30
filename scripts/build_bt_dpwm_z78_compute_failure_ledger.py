"""Build the auditable BT-DPWM parameter, wall-clock, and failure ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import ProjectedResidualInnovation
from robotarm.models.topology_graph_world_model import (
    TopologyGraphConfig, TopologyGraphWorldModel)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z78_compute_ledger_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z78_compute_failure_ledger/summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(
        "config/experiment/g2_bt_dpwm_meta_train_z32_confirmation_z76_v1.yaml"
    ).read_text(encoding="utf-8"))
    adapter_cfg = yaml.safe_load(Path(
        "config/experiment/g2_bt_dpwm_known_topology_z63_v1.yaml"
    ).read_text(encoding="utf-8"))
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(base["baseline_hidden_dim"])))
    bt = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(base["hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=True,
        object_hidden_dim=int(base["object_hidden_dim"]))
    adapter = ProjectedResidualInnovation(
        latent_dim=8, rank=int(adapter_cfg["adapter_rank"]),
        hidden_dim=int(adapter_cfg["adapter_hidden_dim"]),
        position_limit=0.0015, velocity_limit=0.025,
        factorized_context=bool(adapter_cfg["factorized_context"]),
        joint_factorized_basis=bool(adapter_cfg["joint_factorized_basis"]),
        memory_dim=int(adapter_cfg["adapter_memory_dim"]),
        analytic_history=bool(adapter_cfg["analytic_history"]),
        history_deadband=float(adapter_cfg["history_deadband"]),
        shared_joint_basis=bool(adapter_cfg["shared_joint_basis"]))
    encoder = UncertainPhysicalContextEncoder(hidden_dim=96)
    parameters = {"shared_h136": parameter_count(shared),
                  "bt_base": parameter_count(bt),
                  "projected_adapter": parameter_count(adapter),
                  "uncertain_context_encoder": parameter_count(encoder)}
    parameters["shared_deployment_total"] = (
        parameters["shared_h136"]+parameters["projected_adapter"]+
        parameters["uncertain_context_encoder"])
    parameters["bt_deployment_total"] = (
        parameters["bt_base"]+parameters["projected_adapter"]+
        parameters["uncertain_context_encoder"])
    parameters["bt_vs_shared_total_pct"] = 100*(
        parameters["bt_deployment_total"]-parameters["shared_deployment_total"]
    )/parameters["shared_deployment_total"]
    stage_totals = {seed: sum(int(item["wall_clock_s"])
        for item in stages.values()) for seed, stages in cfg["stages"].items()}
    sources = {
        "z71": "runs/g2_bt_dpwm_z71_five_seed/five_seed_gate_v1/summary.json",
        "z75": "runs/g2_bt_dpwm_z75_nested_support/five_seed_development_v1/summary.json",
        "z76": "runs/g2_bt_dpwm_z76_confirmation/two_seed_confirmation_v1/summary.json",
        "z77": "runs/g2_bt_dpwm_z77_robustness/two_seed_summary_v1/summary.json",
    }
    failures = []
    for name, path in sources.items():
        payload = load_json(path)
        failures.append({"run": name, "source": path,
                         "passed": bool(payload["gate"]["passed"]),
                         "gate": payload["gate"]})
    for seed in (57, 67):
        for name, path in (
            ("v0", f"runs/g2_dual_expert_fair_gate_v0/seed{seed}_v1/summary.json"),
            ("z32", f"runs/g2_bt_dpwm_meta_train_z32/seed{seed}_v1/summary.json"),
            ("z69", f"runs/g2_bt_dpwm_zero_topology_columns_z69/seed{seed}_v1/summary.json")):
            payload = load_json(path)
            failures.append({"run": name, "seed": seed, "source": path,
                             "passed": bool(payload["gate_passed"]),
                             "free_improvement_pct": payload.get(
                                 "free_arm_improvement_pct",
                                 payload.get("independent_free_improvement_vs_shared_parameter_matched_pct")),
                             "object_improvement_pct": payload.get(
                                 "object_improvement_pct",
                                 payload.get("independent_object_regression_vs_best_shared_pct")),
                             "overall_improvement_pct": payload.get("overall_improvement_pct")})
    output = {"version": cfg["version"], "hardware": cfg["hardware"],
              "measurement_note": cfg["measurement_note"],
              "parameters": parameters, "wall_clock_stages": cfg["stages"],
              "wall_clock_total_s": stage_totals,
              "known_bottleneck": cfg["known_bottleneck"],
              "failure_runs": failures,
              "all_failures_retained": all(not item["passed"] for item in failures)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": parameters, "wall_clock_total_s": stage_totals,
                      "failure_run_count": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
