from __future__ import annotations

import torch

from robotarm.models.constraint_reaction_world_model import ConstraintReactionWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphWorldModel
from robotarm.models.unconstrained_residual_graph import UnconstrainedResidualGraph


def test_unconstrained_adapter_matches_reaction_trainable_parameter_count() -> None:
    residual = UnconstrainedResidualGraph(TopologyGraphWorldModel())
    reaction = ConstraintReactionWorldModel(TopologyGraphWorldModel())
    residual_count = sum(p.numel() for p in residual.parameters() if p.requires_grad)
    reaction_count = sum(p.numel() for p in reaction.parameters() if p.requires_grad)
    assert residual_count == reaction_count


def test_unconstrained_adapter_shapes_and_gradients() -> None:
    model = UnconstrainedResidualGraph(TopologyGraphWorldModel())
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == (2, 14)
    assert hidden.shape[:2] == (2, 5)
    prediction.pow(2).mean().backward()
    assert model.scale.grad is not None
