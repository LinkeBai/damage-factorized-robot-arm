"""Cheap closed-loop gate for ensemble-mean versus minimax Push MPC."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.planner import PlannerConfig, RobustPushCEMPlanner
from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.g1_mechanism import encode_damage_batch
from robotarm.training.controllers import (
    directional_push_waypoints,
    joint_reference_action,
    solve_reach_reference,
)
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_ensemble import TopologyMember
from scripts.run_push_benchmark import PUSH_XML


def load_ensemble(path: Path, device: torch.device) -> list[TopologyMember]:
    payload = torch.load(path, map_location=device, weights_only=True)
    ensemble = []
    for item in payload:
        encoder = TopologyEncoder().to(device)
        world_model = WorldModel(WorldModelConfig(
            state_dim=item["state_dim"], context_dim=item["context_dim"],
            latent_dim=item.get("latent_dim", 128),
        )).to(device)
        encoder.load_state_dict(item["encoder"])
        world_model.load_state_dict(item["world_model"])
        encoder.eval()
        world_model.eval()
        ensemble.append(TopologyMember(encoder, world_model))
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--approach-steps", type=int, default=30)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument(
        "--target-section", choices=("evaluation", "validation"),
        default="evaluation",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = load_ensemble(args.checkpoint, device)
    protocol = load_g1_protocol(Path("config/splits/g1_push_fewshot_v2.yaml"))
    split = load_target_split(Path("config/splits/push_targets_5dof_v1.yaml"))
    target_items = getattr(split, args.target_section)
    target_item = target_items[args.target_index]
    target = target_item.as_array()
    rows = []
    for domain in protocol.test:
        env = MujocoArmEnv(
            xml_path=PUSH_XML,
            residual_physics=domain.residual,
            block_initial_xy=np.array([0.24, 0.10]),
        )
        ranges = torch.as_tensor(env.joint_ranges, dtype=torch.float32)
        contexts = [
            encode_damage_batch(
                member.encoder, [domain.damage], env.joint_ranges, device
            ).squeeze(0)
            for member in ensemble
        ]
        for risk_alpha, method in (
            (None, "nominal_ik"),
            (0.0, "ensemble_mean"),
            (0.0, "guarded_mean"),
        ):
            observation = env.reset(target=target, damage_config=domain.damage)
            initial = env.block_pos().copy()
            locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
            approach, push_endpoint = directional_push_waypoints(initial, target[:2])
            approach_reference, _ = solve_reach_reference(
                approach, env.joint_ranges, locked_joints=locked
            )
            push_reference, _ = solve_reach_reference(
                push_endpoint,
                env.joint_ranges,
                locked_joints=locked,
            )
            for _ in range(args.approach_steps):
                action = joint_reference_action(
                    observation["state"][:10],
                    approach_reference,
                    locked_joints=tuple(domain.damage.locked),
                )
                observation = env.step(action)["observation"]
            planner = RobustPushCEMPlanner(
                [member.world_model for member in ensemble], contexts,
                PlannerConfig(
                    horizon=args.horizon,
                    candidates=args.candidates,
                    elites=max(2, args.candidates // 4),
                    iterations=2,
                    seed=7,
                ),
                risk_alpha=0.0 if risk_alpha is None else risk_alpha,
            )
            contacts = 0
            fallbacks = 0
            deviations = []
            for _ in range(args.max_steps):
                nominal_action = joint_reference_action(
                    observation["state"][:10],
                    push_reference,
                    locked_joints=tuple(domain.damage.locked),
                )
                if risk_alpha is None:
                    action = nominal_action
                else:
                    action = planner.plan(
                        torch.as_tensor(observation["state"]),
                        torch.as_tensor(target[:2]),
                        ranges,
                        locked_joints=tuple(domain.damage.locked),
                        nominal_action=torch.as_tensor(nominal_action),
                    ).numpy()
                    deviations.append(float(np.linalg.norm(action - nominal_action)))
                    if method == "guarded_mean" and np.linalg.norm(
                        action - nominal_action
                    ) > 0.85:
                        action = nominal_action
                        fallbacks += 1
                transition = env.step(action)
                observation = transition["observation"]
                contacts += int(
                    env.has_contact("tool_geom", "block_geom")
                    or env.has_contact("pusher_geom", "block_geom")
                )
            final = env.block_pos().copy()
            row = {
                "domain": domain.domain_id,
                "target": target_item.target_id,
                "method": method,
                "risk_alpha": risk_alpha,
                "final_distance_m": float(np.linalg.norm(final - target[:2])),
                "block_displacement_m": float(np.linalg.norm(final - initial)),
                "contact_steps": contacts,
                "fallback_steps": fallbacks,
                "mean_action_deviation": float(np.mean(deviations)) if deviations else 0.0,
                "min_action_deviation": float(np.min(deviations)) if deviations else 0.0,
                "max_action_deviation": float(np.max(deviations)) if deviations else 0.0,
            }
            rows.append(row)
            print(row, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "target": target.tolist()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
