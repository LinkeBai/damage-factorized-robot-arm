"""Run the frozen G2 structured-vs-ordinary deep-ensemble comparison."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiment/g2_push_ensemble_v1.yaml"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen in {args.config}")
    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    members = int(config["members"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges
    train = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    ensembles = {}
    train_seconds = {}
    for method, mode in config["condition_modes"].items():
        started = time.perf_counter()
        ensembles[method] = train_topology_ensemble(
            train, ranges, members=members, epochs=epochs, device=device,
            seed=args.seed, condition_mode=mode,
        )
        train_seconds[method] = time.perf_counter() - started
        print(f"{method} trained in {train_seconds[method]:.1f}s", flush=True)

    rows = []
    for index, domain in enumerate(protocol.test):
        trajectories = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        domain_metrics = {}
        for method, mode in config["condition_modes"].items():
            metrics = evaluate_topology_ensemble(
                ensembles[method], domain, trajectories, ranges, device=device,
                horizon=int(config["rollout_horizon"]), condition_mode=mode,
            )
            domain_metrics[method] = metrics
            rows.append({"domain": domain.domain_id, "method": method, **metrics})
        structured = domain_metrics["structured_ensemble"]["ensemble_rmse"]
        ordinary = domain_metrics["ordinary_deep_ensemble"]["ensemble_rmse"]
        improvement = 100.0 * (ordinary - structured) / ordinary
        print(f"{domain.domain_id}: structured vs ordinary {improvement:.2f}%", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config_version": config["version"],
        "config_path": str(args.config),
        "seed": args.seed,
        "members": members,
        "epochs": epochs,
        "steps": steps,
        "device": str(device),
        "protocol_sha256": protocol.sha256,
        "parameters": {
            method: sum(member_parameter_count(member) for member in ensemble)
            for method, ensemble in ensembles.items()
        },
        "train_seconds": train_seconds,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
