"""Post-Q0-B diagnostic: identify what cross-expert discrepancy measures."""
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
from scipy.stats import spearmanr

from robotarm.analysis.dual_expert_risk import partial_spearman
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.dual_expert_world_model import DualExpertWorldModel
from robotarm.models.fixed_transform_graph import FixedTransformGraphConfig, FixedTransformGraphWorldModel
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _contexts, _load_ensemble
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(spearmanr(x, y).statistic)


@torch.no_grad()
def collect_mechanism_records(ensemble, ft_model, domain, trajectories, ranges, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    fusion = DualExpertWorldModel(ensemble, ft_model).to(device).eval()
    records = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        fused_state, hidden = states[:, start].clone(), None
        for depth in range(horizon):
            output, hidden = fusion.step(
                fused_state, actions[:, start + depth], contexts, mask, angle, hidden
            )
            data_mean = output.member_means.mean(dim=0)
            target = states[:, start + depth + 1]
            data_joint_sq = (
                (data_mean[:, :10] - target[:, :10]).pow(2) * free_mask
            ).sum(dim=-1) / free_count
            fusion_joint_sq = (
                (output.mean[:, :10] - target[:, :10]).pow(2) * free_mask
            ).sum(dim=-1) / free_count
            data_joint_error = data_joint_sq.sqrt()
            fusion_joint_error = fusion_joint_sq.sqrt()
            object_error = (output.mean[:, 10:] - target[:, 10:]).pow(2).mean(dim=-1).sqrt()
            overall_error = (output.mean - target).pow(2).mean(dim=-1).sqrt()
            object_epistemic = output.member_means[:, :, 10:].var(
                dim=0, unbiased=False
            ).mean(dim=-1).sqrt()
            for trajectory in range(states.shape[0]):
                records.append({
                    "window_start": start,
                    "trajectory": trajectory,
                    "depth": depth + 1,
                    "cross": float(output.cross_expert_discrepancy[trajectory]),
                    "object_epistemic": float(object_epistemic[trajectory]),
                    "data_joint_error": float(data_joint_error[trajectory]),
                    "fusion_joint_error": float(fusion_joint_error[trajectory]),
                    "correction_gain": float(data_joint_error[trajectory] - fusion_joint_error[trajectory]),
                    "fusion_object_error": float(object_error[trajectory]),
                    "fusion_overall_error": float(overall_error[trajectory]),
                })
            fused_state = output.mean
    return records


def summarize(records):
    targets = [
        "data_joint_error", "correction_gain", "fusion_joint_error",
        "fusion_object_error", "fusion_overall_error",
    ]
    depth_rows = []
    for depth in sorted({item["depth"] for item in records}):
        subset = [item for item in records if item["depth"] == depth]
        cross = np.asarray([item["cross"] for item in subset])
        control = np.asarray([item["object_epistemic"] for item in subset])
        row = {"depth": depth, "n": len(subset)}
        for target in targets:
            values = np.asarray([item[target] for item in subset])
            row[f"spearman_{target}"] = _rank_correlation(cross, values)
            row[f"partial_{target}"] = partial_spearman(cross, values, control)
        depth_rows.append(row)
    result = {"depth_rows": depth_rows}
    for target in targets:
        result[f"mean_depth_spearman_{target}"] = float(np.mean([
            row[f"spearman_{target}"] for row in depth_rows
        ]))
        result[f"mean_depth_partial_{target}"] = float(np.mean([
            row[f"partial_{target}"] for row in depth_rows
        ]))
    result["mean_correction_gain"] = float(np.mean([item["correction_gain"] for item in records]))
    result["positive_correction_fraction"] = float(np.mean([
        item["correction_gain"] > 0 for item in records
    ]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    q0b = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(q0b["q0a_config"]).read_text(encoding="utf-8"))
    if args.seed not in q0b["seeds"]:
        raise ValueError("seed not in diagnostic seed list")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    ensemble = _load_ensemble(args.checkpoint_dir / "ordinary_ensemble.pt", device)
    ft_model = FixedTransformGraphWorldModel(FixedTransformGraphConfig(
        hidden_dim=int(q0a["hidden_dim"])
    )).to(device)
    payload = torch.load(args.checkpoint_dir / "ft_gwm.pt", map_location=device, weights_only=True)
    ft_model.load_state_dict(payload["model_state_dict"])
    domain = next(item for item in protocol.test if item.domain_id == q0b["primary_domain"])
    domain_index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        steps=int(q0a["steps"]), seed=args.seed * 100_000 + domain_index * 1000 + 500,
        targets=evaluation, excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    records = collect_mechanism_records(
        ensemble, ft_model, domain, trajectories, ranges, device,
        int(q0a["rollout_horizon"]),
    )
    summary = {"seed": args.seed, "domain": domain.domain_id, **summarize(records)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(
        f"seed={args.seed} cross->data_joint={summary['mean_depth_partial_data_joint_error']:+.3f} "
        f"cross->gain={summary['mean_depth_partial_correction_gain']:+.3f} "
        f"cross->joint_residual={summary['mean_depth_partial_fusion_joint_error']:+.3f} "
        f"cross->object_residual={summary['mean_depth_partial_fusion_object_error']:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
