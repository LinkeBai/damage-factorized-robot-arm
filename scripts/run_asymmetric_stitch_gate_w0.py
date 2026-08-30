"""W0: zero-training asymmetric subspace stitching from frozen V0 models."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


METHODS = ("shared_compute_matched", "independent_experts", "asymmetric_stitch")


@torch.no_grad()
def evaluate(models, domain, trajectories, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros = torch.zeros_like(mask)
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    horizon = min(horizon, actions.shape[1])
    values = {method: {depth: {"free": [], "object": [], "overall": [], "violation": []}
                       for depth in range(horizon)} for method in METHODS}
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction = {method: states[:, start].clone() for method in METHODS}
        hidden = {
            "shared_baseline": None, "independent_joint": None,
            "independent_object": None, "asymmetric_joint": None,
            "asymmetric_object": None,
        }
        for depth in range(horizon):
            action = actions[:, start + depth]
            target = states[:, start + depth + 1]
            shared, hidden["shared_baseline"] = models["shared_compute_matched"].step(
                prediction["shared_compute_matched"], action, zeros, zeros,
                hidden["shared_baseline"],
            )
            prediction["shared_compute_matched"] = surgery.project_state(shared, mask, angle)

            joint, hidden["independent_joint"] = models["independent_joint"].step(
                prediction["independent_experts"], action, zeros, zeros,
                hidden["independent_joint"],
            )
            obj, hidden["independent_object"] = models["independent_object"].step(
                prediction["independent_experts"], action, zeros, zeros,
                hidden["independent_object"],
            )
            independent = obj.clone(); independent[:, :10] = joint[:, :10]
            prediction["independent_experts"] = surgery.project_state(independent, mask, angle)

            shared_joint, hidden["asymmetric_joint"] = models["shared_compute_matched"].step(
                prediction["asymmetric_stitch"], action, zeros, zeros,
                hidden["asymmetric_joint"],
            )
            specialist_obj, hidden["asymmetric_object"] = models["independent_object"].step(
                prediction["asymmetric_stitch"], action, zeros, zeros,
                hidden["asymmetric_object"],
            )
            asymmetric = specialist_obj.clone(); asymmetric[:, :10] = shared_joint[:, :10]
            prediction["asymmetric_stitch"] = surgery.project_state(asymmetric, mask, angle)

            for method in METHODS:
                error = (prediction[method] - target).pow(2)
                values[method][depth]["free"].append(
                    (error[:, :10] * free_mask).sum(dim=-1) / free_count
                )
                values[method][depth]["object"].append(error[:, 10:].mean(dim=-1))
                values[method][depth]["overall"].append(error.mean(dim=-1))
                values[method][depth]["violation"].append(
                    surgery.constraint_violation(prediction[method], mask, angle).pow(2)
                )
    rows = []
    for method in METHODS:
        for depth in range(horizon):
            rmse = lambda key: float(torch.cat(values[method][depth][key]).mean().sqrt())
            rows.append({
                "method": method, "depth": depth + 1,
                "free_arm_rmse": rmse("free"), "object_rmse": rmse("object"),
                "overall_rmse": rmse("overall"),
                "constraint_violation_rms": rmse("violation"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v0-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen W0 list")
    v0 = yaml.safe_load(Path(config["v0_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    models = {
        "shared_compute_matched": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(v0["shared_compute_matched_hidden_dim"])
        )).to(device),
        "independent_joint": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(v0["independent_hidden_dim"])
        )).to(device),
        "independent_object": TopologyGraphWorldModel(TopologyGraphConfig(
            hidden_dim=int(v0["independent_hidden_dim"])
        )).to(device),
    }
    payload = torch.load(args.v0_run_dir / "models.pt", map_location=device, weights_only=True)
    for name, model in models.items():
        model.load_state_dict(payload[name])
    domain = next(item for item in protocol.test if item.domain_id == config["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        steps=int(q0a["steps"]), seed=args.seed * 100_000 + index * 1000 + 500,
        targets=evaluation, excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    rows = evaluate(models, domain, trajectories, device, int(q0a["rollout_horizon"]))
    final = {row["method"]: row for row in rows if row["depth"] == int(q0a["rollout_horizon"])}
    shared, independent, asymmetric = (
        final["shared_compute_matched"], final["independent_experts"],
        final["asymmetric_stitch"],
    )
    improvement = lambda candidate, baseline, key: 100.0 * (baseline[key] - candidate[key]) / baseline[key]
    comparisons = {
        "free_regression_vs_shared_compute_pct": -improvement(asymmetric, shared, "free_arm_rmse"),
        "object_improvement_vs_shared_compute_pct": improvement(asymmetric, shared, "object_rmse"),
        "overall_improvement_vs_shared_compute_pct": improvement(asymmetric, shared, "overall_rmse"),
        "overall_improvement_vs_independent_pct": improvement(asymmetric, independent, "overall_rmse"),
    }
    gate = config["gate"]
    passed = (
        comparisons["free_regression_vs_shared_compute_pct"]
        <= float(gate["maximum_free_regression_vs_shared_compute_pct"])
        and comparisons["object_improvement_vs_shared_compute_pct"]
        >= float(gate["minimum_object_improvement_vs_shared_compute_pct"])
        and comparisons["overall_improvement_vs_shared_compute_pct"]
        >= float(gate["minimum_overall_improvement_vs_shared_compute_pct"])
        and comparisons["overall_improvement_vs_independent_pct"]
        >= float(gate["minimum_overall_improvement_vs_independent_pct"])
        and asymmetric["constraint_violation_rms"]
        <= float(gate["maximum_constraint_violation_rms"])
    )
    summary = {
        "config_version": config["version"], "seed": args.seed,
        **comparisons, "gate_passed": passed, "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(
        f"[W0] free_reg={comparisons['free_regression_vs_shared_compute_pct']:+.2f}% "
        f"object_imp={comparisons['object_improvement_vs_shared_compute_pct']:+.2f}% "
        f"overall_vs_shared={comparisons['overall_improvement_vs_shared_compute_pct']:+.2f}% "
        f"overall_vs_ind={comparisons['overall_improvement_vs_independent_pct']:+.2f}% "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
