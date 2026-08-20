from __future__ import annotations

import torch

from robotarm.models.reduced_coordinate_graph import ReducedCoordinateGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig


def test_compact_neighbors_bridge_removed_joint() -> None:
    nodes = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    active = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    result = ReducedCoordinateGraphWorldModel._compact_neighbor_sum(nodes, active)
    assert torch.equal(result[:, 0], nodes[:, 1])
    assert torch.equal(result[:, 1], nodes[:, 0] + nodes[:, 3])
    assert torch.equal(result[:, 2], torch.zeros_like(result[:, 2]))
    assert torch.equal(result[:, 3], nodes[:, 1] + nodes[:, 4])
    assert torch.equal(result[:, 4], nodes[:, 3])


def test_compact_edge_features_mark_bridge_span_and_direction() -> None:
    active = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    lock_angle = torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0]])
    features = ReducedCoordinateGraphWorldModel._compact_edge_features(active, lock_angle)
    assert features.shape == (1, 5, 5)
    assert features[0, 1, 1] == 1.0
    assert features[0, 3, 1] == 1.0
    assert features[0, 1, 0] > features[0, 0, 0]
    assert features[0, 2].abs().sum() == 0
    assert features[0, 1, 3] < 0
    assert features[0, 1, 4] > 0


def test_reduced_model_removes_locked_hidden_and_enforces_state() -> None:
    model = ReducedCoordinateGraphWorldModel(TopologyGraphConfig(hidden_dim=128))
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == (2, 14)
    assert hidden.shape == (2, 5, 128)
    assert torch.count_nonzero(hidden[:, 2]) == 0
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))


def test_packed_model_uses_contiguous_active_hidden_slots() -> None:
    model = ReducedCoordinateGraphWorldModel(packed_active_nodes=True)
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 1], angle[:, 1] = 1.0, 0.25
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert torch.count_nonzero(hidden[:, :4]) > 0
    assert torch.count_nonzero(hidden[:, 4]) == 0
    assert torch.allclose(prediction[:, 1], torch.full((2,), 0.25))
    assert torch.allclose(prediction[:, 6], torch.zeros(2))


def test_reduced_model_gradients_exclude_locked_joint_path() -> None:
    model = ReducedCoordinateGraphWorldModel()
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 1], angle[:, 1] = 1.0, 0.5
    prediction, _ = model.step(state, action, mask, angle, None)
    loss = prediction[:, [0, 2, 10]].pow(2).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_detached_object_features_do_not_update_shared_graph() -> None:
    model = ReducedCoordinateGraphWorldModel(detach_object_features=True)
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction[:, 10:].pow(2).mean().backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.node_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.object_head.parameters())
