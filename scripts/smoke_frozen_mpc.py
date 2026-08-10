"""Train a small five-DoF model and smoke-test frozen MPC on D3 Reach."""
from __future__ import annotations

import json
import time
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
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = build_g1_protocol()
    targets = load_target_split()
    calibration_targets = tuple(target.as_array() for target in targets.calibration)
    evaluation_targets = tuple(target.as_array() for target in targets.evaluation[:2])
    ranges = MujocoArmEnv().joint_ranges

    train_data = collect_controller_domains(
        protocol.train,
        trajectories_per_domain=2,
        steps=100,
        seed=70_000,
        targets=calibration_targets,
    )
    models = train_mechanism_models(
        protocol.train,
        train_data,
        ranges,
        epochs=60,
        device=device,
    )
    domain = next(item for item in protocol.test if item.topology == "D3")
    calibration = collect_controller_domains(
        (domain,),
        trajectories_per_domain=5,
        steps=100,
        seed=71_000,
        targets=calibration_targets,
    )
    top_context = topology_only_context(models, domain, ranges, device=device)
    planner = PlannerConfig(
        horizon=5,
        candidates=64,
        elites=8,
        iterations=2,
        seed=seed,
    )
    topology_metrics = evaluate_frozen_mpc(
        models.topology_world_model,
        top_context,
        domain,
        evaluation_targets,
        max_steps=350,
        planner_config=planner,
    )
    dfwm_metrics = {}
    for shots in (0, 1, 2, 5):
        context = infer_dfwm_context(
            models,
            domain,
            calibration,
            ranges,
            shots=shots,
            latent_steps=30,
            device=device,
        )
        dfwm_metrics[f"dfwm_{shots}shot"] = evaluate_frozen_mpc(
            models.dfwm_world_model,
            context,
            domain,
            evaluation_targets,
            max_steps=350,
            planner_config=planner,
        ).as_dict()
    summary = {
        "status": "five_dof_frozen_mpc_smoke_complete",
        "scope": "engineering smoke, not G1 gate evidence",
        "domain": domain.domain_id,
        "actor_or_wm_updated_at_deployment": False,
        "topology_only": topology_metrics.as_dict(),
        **dfwm_metrics,
    }
    run_dir = Path("runs/frozen_mpc_smoke") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
