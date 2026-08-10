from __future__ import annotations

import numpy as np
import torch

from robotarm.models.planner import PlannerConfig
from robotarm.models.world_model import WorldModel
from robotarm.training.control_eval import evaluate_frozen_mpc
from robotarm.training.sim_protocol import DomainSpec


def test_frozen_mpc_evaluation_runs_without_model_update():
    wm = WorldModel()
    before = [parameter.detach().clone() for parameter in wm.parameters()]
    metrics = evaluate_frozen_mpc(
        wm,
        torch.zeros(72),
        DomainSpec("D3", "nominal", "test"),
        (np.array([0.2, 0.0, 0.25]),),
        max_steps=2,
        planner_config=PlannerConfig(
            horizon=2,
            candidates=8,
            elites=2,
            iterations=1,
        ),
    )
    assert metrics.episodes == 1
    assert 0.0 <= metrics.success_rate <= 1.0
    for original, current in zip(before, wm.parameters()):
        assert torch.equal(original, current)
