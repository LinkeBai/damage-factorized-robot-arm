"""Sweep learned-MPC capacity on one fixed G1 target and two damage domains."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.planner import PlannerConfig
from robotarm.training.control_eval import evaluate_frozen_mpc, infer_dfwm_context
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    seed = 17
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = build_g1_protocol()
    split = load_target_split()
    target = split.evaluation[0].as_array()
    ranges = MujocoArmEnv().joint_ranges
    train_data = collect_controller_domains(
        protocol.train, trajectories_per_domain=2, steps=100,
        seed=seed * 10_000, targets=tuple(t.as_array() for t in split.calibration)
    )
    models = train_mechanism_models(protocol.train, train_data, ranges, epochs=60, device=device)
    rows = []
    configs = ((5, 32, 2), (10, 128, 4), (15, 256, 6))
    for domain_index, domain in enumerate(protocol.test):
        calibration = collect_controller_domains(
            (domain,), trajectories_per_domain=5, steps=100,
            seed=seed * 100_000 + domain_index * 1_000,
            targets=tuple(t.as_array() for t in split.calibration)
        )
        for shots in (0, 5):
            context = infer_dfwm_context(
                models, domain, calibration, ranges, shots=shots,
                latent_steps=30, device=device
            )
            for horizon, candidates, iterations in configs:
                metrics = evaluate_frozen_mpc(
                    models.dfwm_world_model, context, domain, (target,),
                    max_steps=300,
                    planner_config=PlannerConfig(
                        horizon=horizon, candidates=candidates,
                        elites=max(4, candidates // 8), iterations=iterations,
                        seed=seed,
                    ),
                )
                rows.append({
                    "seed": seed, "domain": domain.domain_id, "shots": shots,
                    "horizon": horizon, "candidates": candidates,
                    "iterations": iterations, **metrics.as_dict(),
                })
                print(rows[-1], flush=True)
    out = Path("results/final/g1-planner-diagnosis.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
