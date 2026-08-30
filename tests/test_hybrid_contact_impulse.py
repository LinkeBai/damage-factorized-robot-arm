import torch

from robotarm.models.hybrid_contact_impulse import (
    HybridContactImpulseModel,
    oracle_velocity_impulse,
)


def _state(batch: int = 2) -> torch.Tensor:
    state = torch.zeros(batch, 14)
    state[:, 10:12] = torch.tensor([0.20, 0.00])
    return state


def test_non_contact_transition_has_no_impulse() -> None:
    model = HybridContactImpulseModel()
    state = _state()
    prediction, diagnostics = model(state, state[:, :5], torch.zeros(2, dtype=torch.bool))
    assert torch.equal(diagnostics["delta_velocity"], torch.zeros(2, 2))
    assert torch.allclose(prediction[:, :2], state[:, 10:12], atol=1e-7)


def test_impulse_obeys_unilateral_friction_cone() -> None:
    model = HybridContactImpulseModel()
    state = _state(8)
    _, diagnostics = model(state, state[:, :5], torch.ones(8, dtype=torch.bool))
    normal = diagnostics["normal_impulse"]
    tangent = diagnostics["tangent_impulse"]
    assert torch.all(normal >= 0)
    assert torch.all(torch.abs(tangent) <= model.cfg.friction_coefficient * normal + 1e-7)


def test_oracle_velocity_impulse_reconstructs_target_velocity() -> None:
    state = _state()
    state[:, 12:14] = torch.tensor([0.1, -0.2])
    target = state.clone()
    target[:, 12:14] = torch.tensor([0.4, 0.3])
    impulse = oracle_velocity_impulse(state, target)
    assert torch.allclose(state[:, 12:14] + impulse, target[:, 12:14])
