from __future__ import annotations

import torch

from robotarm.models.gated_reaction_graph import GatedReactionGraph
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel


def test_gated_reaction_is_low_capacity_near_zero_and_exact() -> None:
    base = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=128))
    model = GatedReactionGraph(base, bottleneck_dim=16)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 5_000
    assert torch.sigmoid(model.joint_gate_logit) < 0.02
    assert torch.sigmoid(model.object_gate_logit) < 0.02

    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == (2, 14)
    assert hidden.shape == (2, 5, 128)
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))


def test_gated_reaction_head_receives_gradients() -> None:
    model = GatedReactionGraph(TopologyGraphWorldModel())
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 1], angle[:, 1] = 1.0, 0.5
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction.pow(2).mean().backward()
    assert model.joint_gate_logit.grad is not None
    assert model.object_gate_logit.grad is not None
    assert all(parameter.grad is None for parameter in model.base.parameters())
