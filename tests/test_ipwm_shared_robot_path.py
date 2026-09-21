import copy

import pytest
import torch

from scripts.prepare_real_ipwm_trial import equivalent_shared_robot_path
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.models.topology_graph_world_model import TopologyGraphConfig


def make_model():
    torch.manual_seed(20260905)
    intervention = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=8), compact_bridge_object_head=True,
        geometric_object_rank=2, analytic_projection=True,
    ).eval()
    carrier = copy.deepcopy(intervention)
    with torch.no_grad():
        for parameter in carrier.geometric_object_head.parameters():
            parameter.zero_()
    return SelectiveInterventionRollout(intervention, carrier).eval()


@torch.no_grad()
def test_exact_full_state_equivalence_over_50_recurrent_steps_all_fault_masks():
    model = make_model()
    optimized = equivalent_shared_robot_path(model)
    # All 32 binary masks, with independently varying object and robot states.
    mask = torch.tensor([[int(n >> bit & 1) for bit in range(5)]
                         for n in range(32)], dtype=torch.float32)
    initial = torch.randn(32, 14) * 0.1
    initial[:, 10:12] += 0.2
    angles = initial[:, :5].clone()
    old, new = initial.clone(), initial.clone()
    hidden_old = hidden_new = None
    for _ in range(50):
        reference = torch.randn(32, 5) * 0.1
        old_action = (5 * (reference - old[:, :5]) - .5 * old[:, 5:10]).clamp(-1, 1) * (1-mask)
        new_action = (5 * (reference - new[:, :5]) - .5 * new[:, 5:10]).clamp(-1, 1) * (1-mask)
        old, hidden_old = model.step(old, old_action, mask, angles, hidden_old)
        new, hidden_new = optimized.step(new, new_action, mask, angles, hidden_new)
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_rejects_different_robot_weights():
    model = make_model()
    with torch.no_grad():
        next(model.carrier_model.robot_head.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="weights differ"):
        equivalent_shared_robot_path(model)


def test_rejects_object_conditioned_robot():
    model = make_model()
    model.intervention_model.contact_conditioned_robot = True
    with pytest.raises(ValueError, match="object-conditioned"):
        equivalent_shared_robot_path(model)
