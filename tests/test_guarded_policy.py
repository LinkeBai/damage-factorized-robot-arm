import pytest
import torch

from robotarm.models.guarded_policy import guarded_action


def test_rejected_actions_are_exact_carrier_actions():
    torch.manual_seed(7)
    carrier = torch.randn(16, 5)
    candidate = torch.randn(16, 5)
    accept = torch.tensor([True, False] * 8)
    published = guarded_action(candidate, carrier, accept)
    torch.testing.assert_close(published[~accept], carrier[~accept], rtol=0, atol=0)
    torch.testing.assert_close(published[accept], candidate[accept], rtol=0, atol=0)


def test_guard_rejects_ambiguous_shapes_and_non_boolean_decisions():
    with pytest.raises(ValueError):
        guarded_action(torch.zeros(2, 5), torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        guarded_action(torch.zeros(2, 5), torch.zeros(2, 5), torch.zeros(2, 1, dtype=torch.bool))
    with pytest.raises(TypeError):
        guarded_action(torch.zeros(2, 5), torch.zeros(2, 5), torch.zeros(2))
