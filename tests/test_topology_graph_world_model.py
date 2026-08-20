from __future__ import annotations

import torch

from robotarm.models.topology_graph_world_model import TopologyGraphWorldModel


def test_graph_world_model_shapes_and_constraints() -> None:
    model = TopologyGraphWorldModel()
    state = torch.randn(3, 14)
    action = torch.randn(3, 5)
    mask = torch.zeros(3, 5)
    mask[:, 2] = 1.0
    angle = torch.zeros(3, 5)
    angle[:, 2] = -0.5
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == (3, 14)
    assert hidden.shape == (3, 5, model.cfg.hidden_dim)
    assert torch.allclose(prediction[:, 2], torch.full((3,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(3))


def test_graph_world_model_backpropagates() -> None:
    model = TopologyGraphWorldModel()
    state = torch.randn(2, 14)
    action = torch.randn(2, 5)
    mask = torch.zeros(2, 5)
    angle = torch.zeros(2, 5)
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction.pow(2).mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
