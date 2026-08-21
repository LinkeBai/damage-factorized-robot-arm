"""Gate S0: attribute dual-expert gains to constraint-manifold stabilization."""
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
from robotarm.models.fixed_transform_graph import FixedTransformGraphConfig, FixedTransformGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _batch, _contexts, _load_ensemble
from scripts.run_ftgwm_gate_k1 import _train
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


METHODS = (
    "ordinary_autonomous", "ordinary_common_state", "ordinary_direct_projection",
    "matched_joint_product", "matched_projected_product", "ft_product",
)


def _ensemble_step(ensemble, state, action, contexts, hidden):
    means, next_hidden = [], []
    for member, context, member_hidden in zip(ensemble, contexts, hidden):
        output, value = member.world_model.step(state, action, context, member_hidden)
        means.append(output["mean"]); next_hidden.append(value)
    return torch.stack(means).mean(dim=0), next_hidden


@torch.no_grad()
def evaluate_variants(ensemble, ft_model, matched, domain, trajectories, ranges, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    zeros = torch.zeros_like(mask)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    horizon = min(horizon, actions.shape[1])
    accum = {method: {depth: {"free": [], "object": [], "all": [], "violation": []}
                      for depth in range(horizon)} for method in METHODS}

    def record(method, depth, prediction, target):
        error = (prediction - target).pow(2)
        accum[method][depth]["all"].append(error.mean(dim=-1))
        accum[method][depth]["free"].append(
            (error[:, :10] * free_mask).sum(dim=-1) / free_count
        )
        accum[method][depth]["object"].append(error[:, 10:].mean(dim=-1))
        accum[method][depth]["violation"].append(
            surgery.constraint_violation(prediction, mask, angle).pow(2)
        )

    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        member_states = [states[:, start].clone() for _ in ensemble]
        member_hidden = [None] * len(ensemble)
        common = {name: states[:, start].clone() for name in METHODS[1:]}
        common_hidden = {name: [None] * len(ensemble) for name in METHODS[1:]}
        joint_hidden = {"matched_joint_product": None, "matched_projected_product": None, "ft_product": None}
        for depth in range(horizon):
            action = actions[:, start + depth]
            member_means = []
            for index, member in enumerate(ensemble):
                output, member_hidden[index] = member.world_model.step(
                    member_states[index], action, contexts[index], member_hidden[index]
                )
                member_states[index] = output["mean"]
                member_means.append(output["mean"])
            ordinary = torch.stack(member_means).mean(dim=0)
            target = states[:, start + depth + 1]
            record("ordinary_autonomous", depth, ordinary, target)

            mean, common_hidden["ordinary_common_state"] = _ensemble_step(
                ensemble, common["ordinary_common_state"], action, contexts,
                common_hidden["ordinary_common_state"],
            )
            common["ordinary_common_state"] = mean
            record("ordinary_common_state", depth, mean, target)

            projected_input = surgery.project_state(common["ordinary_direct_projection"], mask, angle)
            mean, common_hidden["ordinary_direct_projection"] = _ensemble_step(
                ensemble, projected_input, surgery.project_action(action, mask), contexts,
                common_hidden["ordinary_direct_projection"],
            )
            common["ordinary_direct_projection"] = surgery.project_state(mean, mask, angle)
            record("ordinary_direct_projection", depth, common["ordinary_direct_projection"], target)

            for name, project in (("matched_joint_product", False), ("matched_projected_product", True)):
                state = common[name]
                object_mean, common_hidden[name] = _ensemble_step(
                    ensemble, state, action, contexts, common_hidden[name]
                )
                joint_mean, joint_hidden[name] = matched.step(
                    state, action, zeros, zeros, joint_hidden[name]
                )
                prediction = object_mean.clone(); prediction[:, :10] = joint_mean[:, :10]
                if project:
                    prediction = surgery.project_state(prediction, mask, angle)
                common[name] = prediction
                record(name, depth, prediction, target)

            state = common["ft_product"]
            object_mean, common_hidden["ft_product"] = _ensemble_step(
                ensemble, state, action, contexts, common_hidden["ft_product"]
            )
            joint_mean, joint_hidden["ft_product"] = ft_model.step(
                state, action, mask, angle, joint_hidden["ft_product"]
            )
            prediction = object_mean.clone(); prediction[:, :10] = joint_mean[:, :10]
            common["ft_product"] = prediction
            record("ft_product", depth, prediction, target)

    rows = []
    for method in METHODS:
        for depth in range(horizon):
            values = accum[method][depth]
            rmse = lambda key: float(torch.cat(values[key]).mean().sqrt())
            rows.append({
                "method": method, "depth": depth + 1,
                "overall_rmse": rmse("all"), "free_arm_rmse": rmse("free"),
                "object_rmse": rmse("object"),
                "constraint_violation_rms": rmse("violation"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen stabilization-gate list")
    q0a = yaml.safe_load(Path(config["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    common = dict(
        steps=int(q0a["steps"]), excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=calibration, **common,
    )
    ensemble = _load_ensemble(args.checkpoint_dir / "ordinary_ensemble.pt", device)
    ft_model = FixedTransformGraphWorldModel(FixedTransformGraphConfig(
        hidden_dim=int(q0a["hidden_dim"])
    )).to(device)
    ft_model.load_state_dict(torch.load(
        args.checkpoint_dir / "ft_gwm.pt", map_location=device, weights_only=True
    )["model_state_dict"])
    matched = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(q0a["hidden_dim"])
    )).to(device)
    print("[train] matched unconstrained joint expert", flush=True)
    _train(
        matched, _batch(train, device), epochs=int(config["matched_graph_epochs"]),
        learning_rate=float(config["matched_graph_learning_rate"]), use_topology=False,
        include_object_loss=False, object_loss_weight=0.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": matched.state_dict(), "seed": args.seed}, args.output_dir / "matched_joint.pt")
    domain = next(item for item in protocol.test if item.domain_id == config["primary_domain"])
    index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=args.seed * 100_000 + index * 1000 + 500, targets=evaluation, **common,
    )
    rows = evaluate_variants(
        ensemble, ft_model, matched, domain, trajectories, ranges, device,
        int(q0a["rollout_horizon"]),
    )
    final = {row["method"]: row for row in rows if row["depth"] == int(q0a["rollout_horizon"])}
    pct = lambda candidate, baseline: 100.0 * (baseline - candidate) / baseline
    ft, direct, matched_projected, ordinary = (
        final["ft_product"], final["ordinary_direct_projection"],
        final["matched_projected_product"], final["ordinary_autonomous"],
    )
    free_vs_direct = pct(ft["free_arm_rmse"], direct["free_arm_rmse"])
    free_regression_vs_matched = -pct(ft["free_arm_rmse"], matched_projected["free_arm_rmse"])
    object_regression = -pct(ft["object_rmse"], ordinary["object_rmse"])
    # Teacher-forced proxy: depth-1 gain; autonomous gain: final-depth gain.
    first = {row["method"]: row for row in rows if row["depth"] == 1}
    teacher_gain = pct(first["ft_product"]["free_arm_rmse"], first["ordinary_autonomous"]["free_arm_rmse"])
    autonomous_gain = pct(ft["free_arm_rmse"], ordinary["free_arm_rmse"])
    gain_gap = autonomous_gain - teacher_gain
    gate = config["gate"]
    passed = (
        free_vs_direct >= float(gate["minimum_ft_free_improvement_vs_direct_projection_pct"])
        and free_regression_vs_matched <= float(gate["maximum_ft_free_regression_vs_matched_projected_pct"])
        and object_regression <= float(gate["maximum_ft_object_regression_vs_ordinary_pct"])
        and gain_gap >= float(gate["minimum_autonomous_minus_teacher_forced_gain_pct_points"])
    )
    summary = {
        "config_version": config["version"], "seed": args.seed, "domain": domain.domain_id,
        "ft_free_improvement_vs_direct_projection_pct": free_vs_direct,
        "ft_free_regression_vs_matched_projected_pct": free_regression_vs_matched,
        "ft_object_regression_vs_ordinary_pct": object_regression,
        "depth1_free_gain_vs_ordinary_pct": teacher_gain,
        "depth10_free_gain_vs_ordinary_pct": autonomous_gain,
        "autonomous_minus_depth1_gain_pct_points": gain_gap,
        "gate_passed": passed, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(
        f"[S0] vs_direct={free_vs_direct:+.2f}% vs_matched_reg={free_regression_vs_matched:+.2f}% "
        f"object_reg={object_regression:+.2f}% gain_gap={gain_gap:+.2f}pp "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
