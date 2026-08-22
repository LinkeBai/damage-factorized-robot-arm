import torch

from robotarm.models.physical_context_encoder import (
    PhysicalContextEncoder, UncertainPhysicalContextEncoder)


def test_context_encoder_accepts_variable_transition_budgets():
    encoder = PhysicalContextEncoder()
    mask = torch.zeros(4, 5)
    for budget in (2, 5, 10, 25):
        output = encoder(torch.randn(4, budget + 1, 14),
                         torch.randn(4, budget, 5), mask)
        assert output.shape == (4, 8)


def test_context_encoder_requires_aligned_transitions():
    encoder = PhysicalContextEncoder()
    try:
        encoder(torch.randn(2, 5, 14), torch.randn(2, 5, 5), torch.zeros(2, 5))
    except ValueError:
        return
    raise AssertionError("misaligned states/actions must fail")


def test_uncertain_encoder_returns_finite_diagonal_posterior():
    encoder = UncertainPhysicalContextEncoder(hidden_dim=24)
    mean, log_variance = encoder(
        torch.randn(2, 11, 14), torch.randn(2, 10, 5),
        torch.zeros(2, 5), return_uncertainty=True)
    assert mean.shape == log_variance.shape == (2, 8)
    assert torch.isfinite(mean).all() and torch.isfinite(log_variance).all()
    assert log_variance.min() >= -7.0 and log_variance.max() <= 3.0
