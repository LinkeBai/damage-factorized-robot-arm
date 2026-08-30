from __future__ import annotations

import torch

from robotarm.models.reduced_coordinate_graph import ReducedCoordinateGraphWorldModel
from scripts.diagnose_rcgwm_optimization import _gradient_stats, _loss_parts
from robotarm.training.sim_data import SimTrajectory


def test_diagnostic_loss_and_gradient_stats_are_finite() -> None:
    model = ReducedCoordinateGraphWorldModel()
    states = torch.randn(2, 7, 14)
    actions = torch.randn(2, 6, 5)
    mask = torch.zeros(2, 5)
    angle = torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    free, obj = _loss_parts(model, (states, actions, mask, angle), with_rollout=True)
    shared = list(model.node_encoder.parameters()) + list(model.message.parameters())
    cosine, free_norm, object_norm = _gradient_stats(free, obj, shared)
    values = torch.tensor([
        float(free.detach()), float(obj.detach()), cosine, free_norm, object_norm
    ])
    assert torch.isfinite(values).all()
    assert free_norm > 0 and object_norm > 0
