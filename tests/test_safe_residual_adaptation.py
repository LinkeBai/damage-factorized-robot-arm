import torch

from robotarm.training.safe_residual_adaptation import (
    SafeAdaptConfig, safe_adapt_residual,
)


def test_safe_adaptation_improves_agreeing_fit_and_validation():
    target = torch.tensor([0.2, -0.1])
    loss = lambda z: (z - target.to(z)).pow(2).mean()
    result = safe_adapt_residual(
        loss, loss, device=torch.device("cpu"),
        config=SafeAdaptConfig(latent_dim=2, trust_radius=0.5,
                               minimum_validation_improvement=1e-6),
    )
    assert not result.rolled_back
    assert result.best_validation_loss < result.initial_validation_loss
    assert result.z.norm() <= 0.5 + 1e-6


def test_safe_adaptation_rolls_back_when_validation_disagrees():
    fit = lambda z: (z - 1.0).pow(2).mean()
    validation = lambda z: (z + 1.0).pow(2).mean()
    result = safe_adapt_residual(
        fit, validation, device=torch.device("cpu"),
        config=SafeAdaptConfig(latent_dim=2, validation_tolerance=0.0),
    )
    assert result.rolled_back
    torch.testing.assert_close(result.z, torch.zeros(2))


def test_gradient_normalization_and_trust_region_bound_large_scale_loss():
    loss = lambda z: 1e9 * (z - 10.0).pow(2).mean()
    result = safe_adapt_residual(
        loss, loss, device=torch.device("cpu"),
        config=SafeAdaptConfig(latent_dim=3, trust_radius=0.25,
                               minimum_validation_improvement=1e-9),
    )
    assert torch.isfinite(result.z).all()
    assert result.z.norm() <= 0.25 + 1e-6
