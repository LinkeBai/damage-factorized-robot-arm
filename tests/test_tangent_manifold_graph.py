import torch

from robotarm.models.tangent_manifold_graph import TangentManifoldGraphWorldModel


def test_tangent_model_keeps_locked_temporal_state_zero_and_projects_output():
    model = TangentManifoldGraphWorldModel()
    state = torch.randn(2, 14)
    action = torch.randn(2, 5)
    mask = torch.zeros(2, 5); mask[:, 2] = 1.0
    angle = torch.zeros(2, 5); angle[:, 2] = 0.37
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert torch.allclose(prediction[:, 2], torch.full((2,), 0.37))
    assert torch.all(prediction[:, 7] == 0.0)
    assert torch.all(hidden[:, 2] == 0.0)
    assert torch.allclose(prediction[:, 10:], state[:, 10:])


def test_tangent_model_preserves_full_chain_spatial_nodes():
    model = TangentManifoldGraphWorldModel()
    state = torch.zeros(1, 14)
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    prediction, hidden = model.step(
        state, torch.zeros(1, 5), mask, torch.zeros(1, 5), None
    )
    assert hidden.shape[1] == 5
    assert prediction.shape == state.shape
