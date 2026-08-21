"""U0: isolate product factorization and recurrent projection in DPP-WM."""
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
from scripts.run_dual_expert_gate_q0a import _contexts, _load_ensemble
from scripts.run_manifold_stabilization_gate_s0 import _ensemble_step
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


METHODS = (
    "monolithic_autonomous",
    "monolithic_internal_projection",
    "product_no_projection",
    "product_output_only_projection",
    "dpp_internal_projection",
)


@torch.no_grad()
def evaluate(ensemble, joint_model, domain, trajectories, ranges, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros = torch.zeros_like(mask)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    horizon = min(horizon, actions.shape[1])
    values = {method: {depth: {"free": [], "object": [], "violation": []}
                       for depth in range(horizon)} for method in METHODS}

    def record(method, depth, prediction, target):
        error = (prediction - target).pow(2)
        values[method][depth]["free"].append(
            (error[:, :10] * free_mask).sum(dim=-1) / free_count
        )
        values[method][depth]["object"].append(error[:, 10:].mean(dim=-1))
        values[method][depth]["violation"].append(
            surgery.constraint_violation(prediction, mask, angle).pow(2)
        )

    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        member_states = [states[:, start].clone() for _ in ensemble]
        member_hidden = [None] * len(ensemble)
        mono_projected = states[:, start].clone()
        mono_projected_hidden = [None] * len(ensemble)
        product_state = {
            method: states[:, start].clone() for method in METHODS[2:]
        }
        object_hidden = {method: [None] * len(ensemble) for method in METHODS[2:]}
        joint_hidden = {method: None for method in METHODS[2:]}
        for depth in range(horizon):
            action = actions[:, start + depth]
            target = states[:, start + depth + 1]
            autonomous_means = []
            for index, member in enumerate(ensemble):
                output, member_hidden[index] = member.world_model.step(
                    member_states[index], action, contexts[index], member_hidden[index]
                )
                member_states[index] = output["mean"]
                autonomous_means.append(output["mean"])
            record("monolithic_autonomous", depth, torch.stack(autonomous_means).mean(0), target)

            projected_input = surgery.project_state(mono_projected, mask, angle)
            mono_projected, mono_projected_hidden = _ensemble_step(
                ensemble, projected_input, surgery.project_action(action, mask), contexts,
                mono_projected_hidden,
            )
            mono_projected = surgery.project_state(mono_projected, mask, angle)
            record("monolithic_internal_projection", depth, mono_projected, target)

            for method in METHODS[2:]:
                raw_state = product_state[method]
                object_mean, object_hidden[method] = _ensemble_step(
                    ensemble, raw_state, action, contexts, object_hidden[method]
                )
                joint_mean, joint_hidden[method] = joint_model.step(
                    raw_state, action, zeros, zeros, joint_hidden[method]
                )
                raw_prediction = object_mean.clone()
                raw_prediction[:, :10] = joint_mean[:, :10]
                if method == "dpp_internal_projection":
                    feedback = surgery.project_state(raw_prediction, mask, angle)
                    reported = feedback
                elif method == "product_output_only_projection":
                    feedback = raw_prediction
                    reported = surgery.project_state(raw_prediction, mask, angle)
                else:
                    feedback = reported = raw_prediction
                product_state[method] = feedback
                record(method, depth, reported, target)

    rows = []
    for method in METHODS:
        for depth in range(horizon):
            rmse = lambda key: float(torch.cat(values[method][depth][key]).mean().sqrt())
            rows.append({
                "method": method, "depth": depth + 1,
                "free_arm_rmse": rmse("free"), "object_rmse": rmse("object"),
                "constraint_violation_rms": rmse("violation"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q0a-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--matched-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen U0 list")
    q0a = yaml.safe_load(Path(config["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    ensemble = _load_ensemble(args.q0a_checkpoint_dir / "ordinary_ensemble.pt", device)
    joint_model = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(q0a["hidden_dim"])
    )).to(device)
    joint_model.load_state_dict(torch.load(
        args.matched_checkpoint, map_location=device, weights_only=True
    )["model_state_dict"])
    domain = next(item for item in protocol.test if item.domain_id == config["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        steps=int(q0a["steps"]), seed=args.seed * 100_000 + index * 1000 + 500,
        targets=evaluation, excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    rows = evaluate(
        ensemble, joint_model, domain, trajectories, ranges, device,
        int(q0a["rollout_horizon"]),
    )
    final = {row["method"]: row for row in rows if row["depth"] == int(q0a["rollout_horizon"])}
    dpp = final["dpp_internal_projection"]
    pct = lambda candidate, baseline: 100.0 * (baseline - candidate) / baseline
    comparisons = {
        "dpp_free_improvement_vs_monolithic_internal_projection_pct": pct(
            dpp["free_arm_rmse"], final["monolithic_internal_projection"]["free_arm_rmse"]
        ),
        "dpp_free_improvement_vs_product_no_projection_pct": pct(
            dpp["free_arm_rmse"], final["product_no_projection"]["free_arm_rmse"]
        ),
        "dpp_free_improvement_vs_product_output_only_projection_pct": pct(
            dpp["free_arm_rmse"], final["product_output_only_projection"]["free_arm_rmse"]
        ),
        "dpp_object_regression_vs_monolithic_pct": -pct(
            dpp["object_rmse"], final["monolithic_autonomous"]["object_rmse"]
        ),
    }
    gate = config["gate"]
    passed = (
        comparisons["dpp_free_improvement_vs_monolithic_internal_projection_pct"]
        >= float(gate["minimum_dpp_free_improvement_vs_monolithic_internal_projection_pct"])
        and comparisons["dpp_free_improvement_vs_product_no_projection_pct"]
        >= float(gate["minimum_dpp_free_improvement_vs_product_no_projection_pct"])
        and comparisons["dpp_free_improvement_vs_product_output_only_projection_pct"]
        >= float(gate["minimum_dpp_free_improvement_vs_product_output_only_projection_pct"])
        and comparisons["dpp_object_regression_vs_monolithic_pct"]
        <= float(gate["maximum_object_regression_vs_monolithic_pct"])
        and dpp["constraint_violation_rms"] <= float(gate["maximum_constraint_violation_rms"])
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
        f"[U0] vs_mono={comparisons['dpp_free_improvement_vs_monolithic_internal_projection_pct']:+.2f}% "
        f"vs_no_proj={comparisons['dpp_free_improvement_vs_product_no_projection_pct']:+.2f}% "
        f"vs_output_only={comparisons['dpp_free_improvement_vs_product_output_only_projection_pct']:+.2f}% "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
