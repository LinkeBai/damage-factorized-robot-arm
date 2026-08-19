"""G2 held-out topology experiment.

Trains on D2 + intact only. Evaluates on D3 (held-out topology) and D2
(seen topology control). Three methods:

  structured_ensemble    — correct descriptor at train + test
  ordinary_deep_ensemble — intact descriptor at train + test
  structured_wrong_desc  — correct descriptor at train, intact at test (ablation)

Usage:
  python scripts/run_g2_heldout_topology.py --seed 7 --output-dir runs/g2_heldout_topology/seed7_v1
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiment/g2_push_heldout_topology_v1.yaml"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in frozen seed list {config['seeds']}")

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

    # verify D3 absent from training (protocol integrity check)
    train_topologies = {d.domain_id.split("__")[0] for d in protocol.train}
    if "D3" in train_topologies:
        raise RuntimeError(
            f"Protocol integrity failure: D3 found in training domains. "
            f"This experiment requires D3 to be held out. "
            f"Check {config['protocol']}"
        )
    print(f"Protocol check passed: train topologies = {sorted(train_topologies)}", flush=True)

    # training condition modes (descriptor used during training)
    train_modes = config["condition_modes"]
    # eval condition modes (descriptor used during test; may differ)
    eval_modes = config.get("eval_condition_modes", train_modes)

    # collect training trajectories
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

    # train all methods
    ensembles: dict[str, list] = {}
    train_seconds: dict[str, float] = {}
    for method, mode in train_modes.items():
        print(f"\n[train] {method} (train mode={mode}) …", flush=True)
        t0 = time.perf_counter()
        ensembles[method] = train_topology_ensemble(
            train_trajs, ranges, members=members, epochs=epochs,
            device=device, seed=args.seed, condition_mode=mode,
        )
        train_seconds[method] = time.perf_counter() - t0
        print(f"  done in {train_seconds[method]:.1f}s", flush=True)

    # evaluate on each test domain
    rows = []
    for index, domain in enumerate(protocol.test):
        cond_label = domain.domain_id.split("__")[0]
        print(f"\n[eval] {domain.domain_id} …", flush=True)
        trajectories = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        domain_metrics: dict[str, dict] = {}
        for method in train_modes:
            eval_mode = eval_modes.get(method, train_modes[method])
            metrics = evaluate_topology_ensemble(
                ensembles[method], domain, trajectories, ranges, device=device,
                horizon=int(config["rollout_horizon"]), condition_mode=eval_mode,
            )
            domain_metrics[method] = metrics
            rows.append({
                "domain": domain.domain_id,
                "topology": cond_label,
                "method": method,
                "train_mode": train_modes[method],
                "eval_mode": eval_mode,
                **metrics,
            })
            print(f"  {method} (eval={eval_mode}): rmse={metrics['ensemble_rmse']:.4f}", flush=True)

        # print pairwise comparisons for D3 (the key test)
        if cond_label == "D3" and "structured_ensemble" in domain_metrics and \
                "ordinary_deep_ensemble" in domain_metrics:
            s = domain_metrics["structured_ensemble"]["ensemble_rmse"]
            o = domain_metrics["ordinary_deep_ensemble"]["ensemble_rmse"]
            print(f"  → structured vs ordinary on D3: {100*(o-s)/o:+.2f}%", flush=True)

    # write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
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
        "train_topologies": sorted(train_topologies),
        "parameters": {
            m: sum(member_parameter_count(mb) for mb in ens)
            for m, ens in ensembles.items()
        },
        "train_seconds": train_seconds,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
