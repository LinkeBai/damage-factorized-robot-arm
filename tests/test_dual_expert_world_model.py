import torch
from torch import nn

from robotarm.models.dual_expert_world_model import DualExpertWorldModel
from robotarm.training.topology_ensemble import TopologyMember


class _Encoder(nn.Module):
    def forward(self, value):
        return value


class _Predictive(nn.Module):
    def __init__(self, joint_delta: float, object_delta: float):
        super().__init__()
        self.joint_delta = joint_delta
        self.object_delta = object_delta

    def step(self, state, action, context, hidden):
        mean = state.clone()
        mean[:, :10] += self.joint_delta
        mean[:, 10:] += self.object_delta
        return {"mean": mean}, torch.ones(state.shape[0], 1)


class _Structural(nn.Module):
    def step(self, state, action, mask, lock_angle, hidden):
        prediction = state.clone()
        prediction[:, :10] += 3.0
        return prediction, torch.ones(state.shape[0], 5, 1)


def test_dual_expert_uses_structural_joints_and_mean_predictive_object():
    experts = [
        TopologyMember(_Encoder(), _Predictive(1.0, 2.0)),
        TopologyMember(_Encoder(), _Predictive(2.0, 4.0)),
    ]
    model = DualExpertWorldModel(experts, _Structural())
    state = torch.zeros(2, 14)
    output, hidden = model.step(
        state, torch.zeros(2, 5), [torch.zeros(2, 1)] * 2,
        torch.zeros(2, 5), torch.zeros(2, 5),
    )
    assert torch.all(output.mean[:, :10] == 3.0)
    assert torch.all(output.mean[:, 10:] == 3.0)
    assert torch.allclose(output.cross_expert_discrepancy, torch.full((2,), 1.5))
    assert len(hidden.predictive) == 2
    assert not any(parameter.requires_grad for parameter in model.parameters())
