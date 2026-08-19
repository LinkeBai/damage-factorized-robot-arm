"""Cheap Push screen for topology ensembles and calibrated disagreement."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_ensemble import (
    TopologyMember,
    evaluate_topology_ensemble,
    train_topology_ensemble,
)
from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel, WorldModelConfig
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path("config/splits/g1_push_fewshot_v2.yaml"))
    target_split = load_target_split(Path("config/splits/push_targets_5dof_v1.yaml"))
    calibration = tuple(item.as_array() for item in target_split.calibration)
    evaluation = tuple(item.as_array() for item in target_split.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=2, steps=args.steps,
        seed=args.seed * 10_000, targets=calibration, excitation="goal",
        block_initial_xy=np.array([0.24, 0.10]),
    )
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        ensemble = []
        for item in payload:
            encoder = TopologyEncoder().to(device)
            world_model = WorldModel(WorldModelConfig(
                state_dim=item["state_dim"], context_dim=item["context_dim"]
            )).to(device)
            encoder.load_state_dict(item["encoder"])
            world_model.load_state_dict(item["world_model"])
            ensemble.append(TopologyMember(encoder, world_model))
        print("ensemble loaded", flush=True)
    else:
        ensemble = train_topology_ensemble(
            train, ranges, members=args.members, epochs=args.epochs,
            device=device, seed=args.seed,
        )
        print("ensemble trained", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        torch.save(
        [
            {
                "encoder": member.encoder.state_dict(),
                "world_model": member.world_model.state_dict(),
                "state_dim": member.world_model.cfg.state_dim,
                "context_dim": member.world_model.cfg.context_dim,
            }
            for member in ensemble
        ],
            args.output_dir / "ensemble.pt",
        )
    rows = []
    for index, domain in enumerate(protocol.test):
        trajectories = collect_push_domains(
            (domain,), trajectories_per_domain=3, steps=args.steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation, excitation="goal",
            block_initial_xy=np.array([0.24, 0.10]),
        )
        metrics = evaluate_topology_ensemble(
            ensemble, domain, trajectories, ranges, device=device,
        )
        rows.append({"domain": domain.domain_id, **metrics})
        print(f"{domain.domain_id}: {metrics}", flush=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": args.seed, "members": args.members, "epochs": args.epochs,
        "steps": args.steps, "device": str(device), "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
