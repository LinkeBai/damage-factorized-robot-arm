import torch

from robotarm.models.residual_correction import ResidualCorrection


def test_residual_correction_starts_as_zero_delta():
    model = ResidualCorrection(state_dim=14, action_dim=5)
    output = model(torch.randn(3, 14), torch.randn(3, 5), torch.randn(3, 8))
    assert output.shape == (3, 14)
    assert torch.allclose(output, torch.zeros_like(output))


def test_residual_correction_uses_separate_group_gates():
    model = ResidualCorrection(state_dim=14, action_dim=5)
    assert model.arm_delta.out_features == 10
    assert model.object_delta.out_features == 4
    assert model.gate.out_features == 2
