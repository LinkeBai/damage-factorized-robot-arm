"""Q0-A: frozen product-space fusion fidelity gate."""
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
from robotarm.models.dual_expert_world_model import DualExpertWorldModel
from robotarm.models.fixed_transform_graph import (
    FixedTransformGraphConfig,
    FixedTransformGraphWorldModel,
)
from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.g1_mechanism import TOPOLOGY_DIM, encode_damage_batch
from robotarm.training.sim_protocol import damage_from_name, load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_ensemble import (
    TopologyMember,
    conditioning_damages,
    train_topology_ensemble,
)
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_ftgwm_gate_k1 import _batch, _train
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def _save_ensemble(ensemble: list[TopologyMember], path: Path) -> None:
    torch.save([
        {
            "encoder": member.encoder.state_dict(),
            "world_model": member.world_model.state_dict(),
            "state_dim": member.world_model.cfg.state_dim,
            "context_dim": member.world_model.cfg.context_dim,
            "latent_dim": member.world_model.cfg.latent_dim,
        }
        for member in ensemble
    ], path)


def _load_ensemble(path: Path, device: torch.device) -> list[TopologyMember]:
    payload = torch.load(path, map_location=device, weights_only=True)
    ensemble = []
    for item in payload:
        encoder = TopologyEncoder().to(device)
        model = WorldModel(WorldModelConfig(
            state_dim=item["state_dim"], context_dim=item["context_dim"],
            latent_dim=item.get("latent_dim", 128),
        )).to(device)
        encoder.load_state_dict(item["encoder"])
        model.load_state_dict(item["world_model"])
        ensemble.append(TopologyMember(encoder, model))
    return ensemble


def _contexts(ensemble, batch_size, ranges, device):
    damages = conditioning_damages(
        [damage_from_name("intact")] * batch_size, "constant"
    )
    return [
        encode_damage_batch(member.encoder, damages, ranges, device)
        for member in ensemble
    ]


@torch.no_grad()
def _evaluate(ensemble, ft_model, domain, trajectories, ranges, device, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    contexts = _contexts(ensemble, len(trajectories), ranges, device)
    fusion = DualExpertWorldModel(ensemble, ft_model).to(device).eval()
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    values = {
        "ordinary_all": [], "ordinary_free": [], "ordinary_object": [],
        "fusion_all": [], "fusion_free": [], "fusion_object": [],
        "violation": [], "epistemic": [], "cross": [],
    }
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        ordinary_states = [states[:, start].clone() for _ in ensemble]
        ordinary_hidden = [None] * len(ensemble)
        fused_state = states[:, start].clone()
        fused_hidden = None
        for offset in range(horizon):
            member_predictions = []
            for index, member in enumerate(ensemble):
                output, ordinary_hidden[index] = member.world_model.step(
                    ordinary_states[index], actions[:, start + offset],
                    contexts[index], ordinary_hidden[index],
                )
                ordinary_states[index] = output["mean"]
                member_predictions.append(output["mean"])
            ordinary = torch.stack(member_predictions).mean(dim=0)
            fused, fused_hidden = fusion.step(
                fused_state, actions[:, start + offset], contexts, mask, angle,
                fused_hidden,
            )
            fused_state = fused.mean
            target = states[:, start + offset + 1]
            for prefix, prediction in (("ordinary", ordinary), ("fusion", fused.mean)):
                error = (prediction - target).pow(2)
                values[f"{prefix}_all"].append(error.mean(dim=-1))
                values[f"{prefix}_free"].append(
                    (error[:, :10] * free_mask).sum(dim=-1) / free_count
                )
                values[f"{prefix}_object"].append(error[:, 10:].mean(dim=-1))
            values["violation"].append(
                surgery.constraint_violation(fused.mean, mask, angle).pow(2)
            )
            values["epistemic"].append(fused.epistemic_uncertainty)
            values["cross"].append(fused.cross_expert_discrepancy)

    rmse = lambda key: float(torch.stack(values[key], dim=1).mean().sqrt())
    mean = lambda key: float(torch.stack(values[key], dim=1).mean())
    return {
        "ordinary_overall_rmse": rmse("ordinary_all"),
        "ordinary_free_arm_rmse": rmse("ordinary_free"),
        "ordinary_object_rmse": rmse("ordinary_object"),
        "fusion_overall_rmse": rmse("fusion_all"),
        "fusion_free_arm_rmse": rmse("fusion_free"),
        "fusion_object_rmse": rmse("fusion_object"),
        "fusion_constraint_violation_rms": rmse("violation"),
        "mean_epistemic_uncertainty": mean("epistemic"),
        "mean_cross_expert_discrepancy": mean("cross"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ordinary-checkpoint", type=Path)
    parser.add_argument("--ft-checkpoint", type=Path)
    parser.add_argument("--ordinary-epochs", type=int)
    parser.add_argument("--ft-epochs", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    allowed_seeds = config["seeds"] + config.get("extension_seeds", [])
    if args.seed not in allowed_seeds:
        raise ValueError("seed not in frozen list")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    steps = args.steps or int(config["steps"])
    common = dict(
        steps=steps, excitation="goal",
        block_initial_xy=np.asarray(config["block_initial_xy"], dtype=float),
        goal_exploration_std=float(config["goal_exploration_std"]),
    )
    train = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=calibration, **common,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.ordinary_checkpoint:
        ensemble = _load_ensemble(args.ordinary_checkpoint, device)
    else:
        ensemble = train_topology_ensemble(
            train, ranges, members=int(config["ordinary_members"]),
            epochs=args.ordinary_epochs or int(config["ordinary_epochs"]),
            device=device, seed=args.seed, condition_mode="constant",
        )
        _save_ensemble(ensemble, args.output_dir / "ordinary_ensemble.pt")
    ft_model = FixedTransformGraphWorldModel(FixedTransformGraphConfig(
        hidden_dim=int(config["hidden_dim"])
    )).to(device)
    if args.ft_checkpoint:
        payload = torch.load(args.ft_checkpoint, map_location=device, weights_only=True)
        ft_model.load_state_dict(payload["model_state_dict"])
    else:
        _train(
            ft_model, _batch(train, device),
            epochs=args.ft_epochs or int(config["ft_epochs"]),
            learning_rate=float(config["ft_learning_rate"]), use_topology=True,
            include_object_loss=False, object_loss_weight=0.0,
        )
        torch.save({
            "model_state_dict": ft_model.state_dict(), "model_type": "joint_only",
            "config": {"hidden_dim": int(config["hidden_dim"])},
            "seed": args.seed, "protocol_sha256": protocol.sha256,
        }, args.output_dir / "ft_gwm.pt")

    rows = []
    for index, domain in enumerate(protocol.test):
        trajectories = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation, **common,
        )
        metrics = _evaluate(
            ensemble, ft_model, domain, trajectories, ranges, device,
            int(config["rollout_horizon"]),
        )
        rows.append({"domain": domain.domain_id, **metrics})
        print(
            f"{domain.domain_id}: ordinary_obj={metrics['ordinary_object_rmse']:.4f} "
            f"fusion_obj={metrics['fusion_object_rmse']:.4f} "
            f"fusion_free={metrics['fusion_free_arm_rmse']:.4f}", flush=True,
        )
    primary = next(row for row in rows if row["domain"] == config["primary_domain"])
    pct = lambda fused, base: 100.0 * (fused - base) / base
    object_regression = pct(primary["fusion_object_rmse"], primary["ordinary_object_rmse"])
    free_regression = pct(primary["fusion_free_arm_rmse"], primary["ordinary_free_arm_rmse"])
    gate = config["gate"]
    passed = (
        object_regression <= float(gate["maximum_object_rmse_regression_pct"])
        and free_regression <= float(gate["maximum_free_arm_rmse_regression_pct"])
        and primary["fusion_constraint_violation_rms"]
        <= float(gate["maximum_constraint_violation_rms"])
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "config_version": config["version"], "seed": args.seed,
        "protocol_sha256": protocol.sha256, "device": str(device),
        "object_regression_pct": object_regression,
        "free_arm_regression_pct": free_regression,
        "gate_passed": passed, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[Q0-A] object={object_regression:+.2f}% free={free_regression:+.2f}% "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
