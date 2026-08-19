import torch

from robotarm.models.residual_film import ResidualFiLMWorldModel
from robotarm.models.world_model import WorldModel, WorldModelConfig


def test_zero_initialized_film_matches_base_model():
    base = WorldModel(WorldModelConfig(state_dim=14, context_dim=64))
    film = ResidualFiLMWorldModel(base, rank=4)
    state = torch.randn(3, 14)
    action = torch.randn(3, 5)
    topology = torch.randn(3, 64)
    residual = torch.randn(3, 8)
    expected, _ = base.step(state, action, topology, None)
    actual, _ = film.step(state, action, topology, residual, None)
    assert torch.allclose(actual["mean"], expected["mean"])
    assert torch.allclose(actual["log_std"], expected["log_std"])


def test_film_condition_uses_state_action_and_residual():
    base = WorldModel(WorldModelConfig(state_dim=14, action_dim=5, context_dim=64))
    film = ResidualFiLMWorldModel(base, residual_dim=8, rank=4)
    assert film.condition_encoder[0].in_features == 14 + 5 + 8
