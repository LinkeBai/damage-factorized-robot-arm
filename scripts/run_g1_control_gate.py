"""Run the three-seed frozen-MPC G1 control gate after the data pivot."""
from __future__ import annotations

import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.planner import PlannerConfig
from robotarm.training.control_eval import (
    evaluate_frozen_mpc,
    infer_dfwm_context,
    topology_only_context,
)
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    shots_set = (0, 1, 2, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = build_g1_protocol()
    split = load_target_split()
    calibration_targets = tuple(target.as_array() for target in split.calibration)
    evaluation_targets = tuple(target.as_array() for target in split.evaluation[:1])
    ranges = MujocoArmEnv().joint_ranges
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    run_dir = Path("runs/g1_control_gate") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_data = collect_controller_domains(
            protocol.train,
            trajectories_per_domain=2,
            steps=100,
            seed=seed * 10_000,
            targets=calibration_targets,
        )
        models = train_mechanism_models(
            protocol.train, train_data, ranges, epochs=60, device=device
        )
        for domain_index, domain in enumerate(protocol.test):
            calibration = collect_controller_domains(
                (domain,),
                trajectories_per_domain=5,
                steps=100,
                seed=seed * 100_000 + domain_index * 1_000,
                targets=calibration_targets,
            )
            planner = PlannerConfig(
                horizon=5,
                candidates=args.candidates,
                elites=max(4, args.candidates // 8),
                iterations=2,
                seed=seed,
            )
            top_context = topology_only_context(
                models, domain, ranges, device=device
            )
            top = evaluate_frozen_mpc(
                models.topology_world_model,
                top_context,
                domain,
                evaluation_targets,
                max_steps=args.max_steps,
                planner_config=planner,
            )
            rows.append(
                {"seed": seed, "domain": domain.domain_id, "model": "topology_only", "shots": 0, **top.as_dict()}
            )
            for shots in shots_set:
                context = infer_dfwm_context(
                    models,
                    domain,
                    calibration,
                    ranges,
                    shots=shots,
                    latent_steps=30,
                    device=device,
                )
                metrics = evaluate_frozen_mpc(
                    models.dfwm_world_model,
                    context,
                    domain,
                    evaluation_targets,
                    max_steps=args.max_steps,
                    planner_config=planner,
                )
                rows.append(
                    {"seed": seed, "domain": domain.domain_id, "model": "dfwm", "shots": shots, **metrics.as_dict()}
                )
        with (run_dir / "control_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"seed={seed} complete elapsed={time.perf_counter()-started:.1f}s", flush=True)

    summary = {
        "status": "g1_control_gate_complete",
        "seeds": list(seeds),
        "domains": [domain.domain_id for domain in protocol.test],
        "shots": list(shots_set),
        "evaluation_targets": [target.target_id for target in split.evaluation[:1]],
        "actor_or_world_model_updated_at_deployment": False,
        "data_policy": "controller_induced",
        "multi_step_training_horizon": 5,
        "planner_candidates": args.candidates,
        "max_steps": args.max_steps,
        "wall_clock_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
