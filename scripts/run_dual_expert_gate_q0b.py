"""Q0-B: fixed-depth conditional-risk gate for dual experts."""
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

from robotarm.analysis.dual_expert_risk import fixed_depth_risk_summary
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.dual_expert_world_model import DualExpertWorldModel
from robotarm.models.fixed_transform_graph import (
    FixedTransformGraphConfig,
    FixedTransformGraphWorldModel,
)
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_dual_expert_gate_q0a import _contexts, _load_ensemble
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


@torch.no_grad()
def collect_records(
    ensemble, ft_model, domain, trajectories, ranges, device, horizon
) -> list[dict[str, float]]:
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    fusion = DualExpertWorldModel(ensemble, ft_model).to(device).eval()
    records = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction = states[:, start].clone()
        hidden = None
        for depth in range(horizon):
            output, hidden = fusion.step(
                prediction, actions[:, start + depth], contexts, mask, angle, hidden
            )
            prediction = output.mean
            target = states[:, start + depth + 1]
            error = (prediction - target).pow(2).mean(dim=-1).sqrt()
            object_epistemic = output.member_means[:, :, 10:].var(
                dim=0, unbiased=False
            ).mean(dim=-1).sqrt()
            for trajectory_index in range(states.shape[0]):
                records.append({
                    "window_start": start,
                    "trajectory": trajectory_index,
                    "depth": depth,
                    "object_epistemic": float(object_epistemic[trajectory_index]),
                    "cross": float(output.cross_expert_discrepancy[trajectory_index]),
                    "error": float(error[trajectory_index]),
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen list")
    q0a_path = Path(config["q0a_config"])
    q0a = yaml.safe_load(q0a_path.read_text(encoding="utf-8"))
    ordinary_checkpoint = args.checkpoint_dir / "ordinary_ensemble.pt"
    ft_checkpoint = args.checkpoint_dir / "ft_gwm.pt"
    if not ordinary_checkpoint.exists() or not ft_checkpoint.exists():
        raise FileNotFoundError("Q0-A ordinary_ensemble.pt and ft_gwm.pt are required")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    ensemble = _load_ensemble(ordinary_checkpoint, device)
    ft_model = FixedTransformGraphWorldModel(FixedTransformGraphConfig(
        hidden_dim=int(q0a["hidden_dim"])
    )).to(device)
    ft_payload = torch.load(ft_checkpoint, map_location=device, weights_only=True)
    ft_model.load_state_dict(ft_payload["model_state_dict"])
    domain = next(
        item for item in protocol.test if item.domain_id == config["primary_domain"]
    )
    domain_index = list(protocol.test).index(domain)
    trajectories = collect_push_domains(
        (domain,),
        trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        steps=int(q0a["steps"]),
        seed=args.seed * 100_000 + domain_index * 1000 + 500,
        targets=evaluation, excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    records = collect_records(
        ensemble, ft_model, domain, trajectories, ranges, device,
        int(q0a["rollout_horizon"]),
    )
    summary = fixed_depth_risk_summary(records, list(config["coverages"]))
    gate = config["gate"]
    passed = (
        summary["mean_aurc_improvement_pct"]
        >= float(gate["minimum_mean_aurc_improvement_pct"])
        and summary["mean_partial_spearman"]
        > float(gate["minimum_mean_partial_spearman"])
    )
    result = {
        "config_version": config["version"], "seed": args.seed,
        "domain": domain.domain_id, "device": str(device),
        "protocol_sha256": protocol.sha256, "gate_passed": passed,
        **summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(
        f"[Q0-B] seed={args.seed} AURC={summary['mean_aurc_improvement_pct']:+.2f}% "
        f"partial_r={summary['mean_partial_spearman']:+.3f} "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
