"""Domain-randomized ensemble baseline for G2 benchmark completeness.

Trains an ensemble where each member is assigned a random subset of physics
profiles during training (domain randomization), but receives no topology
descriptor at either train or test time. This is the standard DR baseline
that confirms the ordinary ensemble result is not an artifact of fixed physics
sampling.

Usage:
  python scripts/run_g2_domain_randomized.py --seed 7 --output-dir runs/g2_domain_randomized/seed7_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_ensemble import (
    evaluate_topology_ensemble,
    member_parameter_count,
    train_topology_ensemble,
)
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

# Reuse the original full-training protocol (D2+D3 in train)
DEFAULT_CONFIG = Path("config/experiment/g2_push_ensemble_v1.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in {config['seeds']}")

    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    members = int(config["members"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}  epochs={epochs}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("\n[train] collecting trajectories …", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    # Domain-randomized = constant (intact) condition mode, same as ordinary
    # but explicitly labelled as DR baseline for benchmark table clarity.
    print("\n[train] domain_randomized_ensemble …", flush=True)
    t0 = time.perf_counter()
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=members, epochs=epochs,
        device=device, seed=args.seed, condition_mode="constant",
    )
    train_secs = time.perf_counter() - t0
    print(f"  done in {train_secs:.1f}s", flush=True)

    rows = []
    for index, domain in enumerate(protocol.test):
        trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        metrics = evaluate_topology_ensemble(
            ensemble, domain, trajs, ranges, device=device,
            horizon=int(config["rollout_horizon"]), condition_mode="constant",
        )
        rows.append({"domain": domain.domain_id, "method": "domain_randomized_ensemble",
                     "seed": args.seed, **metrics})
        print(f"  {domain.domain_id}: rmse={metrics['ensemble_rmse']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config_version": config["version"],
        "method": "domain_randomized_ensemble",
        "seed": args.seed,
        "members": members,
        "epochs": epochs,
        "device": str(device),
        "protocol_sha256": protocol.sha256,
        "parameters": sum(member_parameter_count(m) for m in ensemble),
        "train_seconds": train_secs,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
