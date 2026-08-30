"""G2 member-count ablation: 1 / 3 / 5 ensemble members.

Reuses the frozen G2 protocol (g2_push_ensemble_v1.yaml) but overrides
the member count. Results go to:
  runs/g2_member_ablation/members{N}/seed{seed}_v1/

Usage:
  python scripts/run_g2_member_ablation.py --seed 7 --members 1
  python scripts/run_g2_member_ablation.py --seed 7 --members 3
  python scripts/run_g2_member_ablation.py --seed 7 --members 5
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

from src.robotarm.envs.mujoco_env import MujocoArmEnv
from src.robotarm.training.sim_protocol import load_g1_protocol
from src.robotarm.training.target_split import load_target_split
from src.robotarm.training.topology_ensemble import (
    evaluate_topology_ensemble,
    member_parameter_count,
    train_topology_ensemble,
)
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_push_ensemble_v1.yaml")
ABLATION_MEMBERS = [1, 3, 5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--members", type=int, required=True, choices=ABLATION_MEMBERS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in frozen seeds {config['seeds']}")

    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    members = args.members
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)

    output_dir = args.output_dir or (
        Path("runs/g2_member_ablation") / f"members{members}" / f"seed{args.seed}_v1"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"members={members}  seed={args.seed}  epochs={epochs}  device={device}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("[train] collecting trajectories …", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    # ordinary ensemble (constant condition mode) with ablated member count
    print(f"[train] ordinary_ensemble members={members} …", flush=True)
    t0 = time.perf_counter()
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=members, epochs=epochs,
        device=device, seed=args.seed, condition_mode="constant",
    )
    train_secs = time.perf_counter() - t0
    print(f"  done in {train_secs:.1f}s", flush=True)

    rows = []
    for index, domain in enumerate(protocol.test):
        test_trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        metrics = evaluate_topology_ensemble(
            ensemble, domain, test_trajs, ranges, device=device,
            horizon=int(config["rollout_horizon"]), condition_mode="constant",
        )
        rows.append({
            "domain": domain.domain_id,
            "method": f"ordinary_ensemble_m{members}",
            "members": members,
            "seed": args.seed,
            **metrics,
        })
        print(f"  {domain.domain_id}: rmse={metrics['ensemble_rmse']:.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config_version": config["version"],
        "ablation": "member_count",
        "members": members,
        "seed": args.seed,
        "epochs": epochs,
        "device": str(device),
        "protocol_sha256": protocol.sha256,
        "parameters": sum(member_parameter_count(m) for m in ensemble),
        "train_seconds": train_secs,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
