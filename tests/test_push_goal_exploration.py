from __future__ import annotations

import numpy as np

from robotarm.training.sim_protocol import DomainSpec
from scripts.run_push_benchmark import collect_push_trajectory


def test_goal_exploration_is_seeded_bounded_and_preserves_lock() -> None:
    domain = DomainSpec("D3", "nominal", "train")
    kwargs = dict(
        domain=domain, steps=12, target=np.array([0.20, 0.10, 0.025]),
        excitation="goal", goal_exploration_std=0.08,
        block_initial_xy=np.array([0.24, 0.10]),
    )
    first = collect_push_trajectory(seed=101, **kwargs)
    repeated = collect_push_trajectory(seed=101, **kwargs)
    different = collect_push_trajectory(seed=202, **kwargs)
    assert first.actions.equal(repeated.actions)
    assert not first.actions.equal(different.actions)
    assert float(first.actions.abs().max()) <= 1.0
    assert float(first.actions[:, 2].abs().max()) == 0.0
