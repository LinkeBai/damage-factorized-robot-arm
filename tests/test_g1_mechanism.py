from __future__ import annotations

import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.g1_mechanism import (
    evaluate_test_domain,
    train_mechanism_models,
)
from robotarm.training.sim_data import collect_domains
from robotarm.training.sim_protocol import DomainSpec


def test_g1_mechanism_minimal_train_and_few_shot_eval():
    train_domains = (
        DomainSpec("intact", "nominal", "train"),
        DomainSpec("D2", "weak_motor", "train"),
    )
    train_data = collect_domains(
        train_domains,
        trajectories_per_domain=1,
        steps=4,
        seed=10,
    )
    ranges = MujocoArmEnv().joint_ranges
    models = train_mechanism_models(
        train_domains,
        train_data,
        ranges,
        epochs=2,
        device=torch.device("cpu"),
    )
    test_domain = DomainSpec("D3", "mixed_unseen", "test")
    calibration = collect_domains(
        (test_domain,),
        trajectories_per_domain=5,
        steps=4,
        seed=20,
    )
    evaluation = collect_domains(
        (test_domain,),
        trajectories_per_domain=1,
        steps=4,
        seed=40,
    )
    rows = evaluate_test_domain(
        models,
        test_domain,
        calibration,
        evaluation,
        ranges,
        shots=(0, 1),
        latent_steps=2,
        device=torch.device("cpu"),
    )
    assert len(rows) == 12
    assert {row["model"] for row in rows} == {
        "topology_only",
        "history_encoder",
        "parameter_matched",
        "residual_only",
        "monolithic_matched",
        "dfwm",
    }
    assert all(torch.isfinite(torch.tensor(row["eval_nll"])) for row in rows)
