from __future__ import annotations

import torch

from robotarm.models.constraint_reaction_world_model import ConstraintReactionWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphWorldModel


def test_reaction_model_freezes_base_and_enforces_lock() -> None:
    base = TopologyGraphWorldModel()
    model = ConstraintReactionWorldModel(base)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == (2, 14)
    assert hidden.shape[:2] == (2, 5)
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))


def test_reaction_adapter_receives_gradients() -> None:
    model = ConstraintReactionWorldModel(TopologyGraphWorldModel())
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 1], angle[:, 1] = 1.0, 0.5
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction.pow(2).mean().backward()
    assert model.scale.grad is not None
    assert all(parameter.grad is None for parameter in model.base.parameters())
