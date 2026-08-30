import torch

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.projected_residual_innovation import (
    FewShotProjectedModel,
    ProjectedResidualInnovation,
)


def _inputs(batch=3):
    return (torch.randn(batch, 14), torch.randn(batch, 5),
            torch.zeros(batch, 5), torch.zeros(batch, 5))


def test_zero_context_is_exactly_zero_without_training_assumptions():
    adapter = ProjectedResidualInnovation()
    state, action, mask, _ = _inputs()
    correction = adapter(state, action, mask, torch.zeros(8))
    torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_adapter_only_changes_free_robot_coordinates():
    adapter = ProjectedResidualInnovation()
    state, action, mask, _ = _inputs()
    mask[:, 2] = 1.0
    correction = adapter(state, action, mask, torch.randn(8))
    torch.testing.assert_close(correction[:, 2], torch.zeros(3))
    torch.testing.assert_close(correction[:, 7], torch.zeros(3))
    torch.testing.assert_close(correction[:, 10:], torch.zeros(3, 4))
    assert torch.count_nonzero(correction[:, :10]) > 0


def test_wrapper_recovers_base_at_k0_and_projects_locked_joint():
    torch.manual_seed(5)
    base = BlockTriangularDPWM()
    wrapped = FewShotProjectedModel(base, ProjectedResidualInnovation())
    state, action, mask, angle = _inputs()
    mask[:, 1], angle[:, 1] = 1.0, 0.23
    base_prediction, _ = base.step(state, action, mask, angle, None)
    wrapped_prediction, _ = wrapped.step(state, action, mask, angle, None)
    torch.testing.assert_close(wrapped_prediction, base_prediction)
    wrapped.set_residual_context(torch.randn(8))
    adapted_prediction, _ = wrapped.step(state, action, mask, angle, None)
    torch.testing.assert_close(adapted_prediction[:, 1], angle[:, 1])
    torch.testing.assert_close(adapted_prediction[:, 6], torch.zeros(3))
    assert not torch.allclose(adapted_prediction[:, 10:], base_prediction[:, 10:])


def test_bt_object_block_receives_adapted_robot_transition():
    torch.manual_seed(9)
    base = BlockTriangularDPWM(independent_object_encoder=True)
    wrapped = FewShotProjectedModel(base, ProjectedResidualInnovation())
    state, action, mask, angle = _inputs()
    zero_prediction, _ = wrapped.step(state, action, mask, angle, None)
    wrapped.set_residual_context(torch.ones(8))
    adapted_prediction, _ = wrapped.step(state, action, mask, angle, None)
    assert not torch.allclose(adapted_prediction[:, :10], zero_prediction[:, :10])
    assert not torch.allclose(adapted_prediction[:, 10:], zero_prediction[:, 10:])


def test_post_object_ablation_blocks_residual_from_object_transition():
    torch.manual_seed(13)
    base = BlockTriangularDPWM(independent_object_encoder=True)
    adapter = ProjectedResidualInnovation(latent_dim=8, rank=8)
    wrapped = FewShotProjectedModel(base, adapter, adapter_before_object=False)
    state, action, mask, angle = _inputs()
    wrapped.set_residual_context(torch.zeros(8))
    zero_prediction, _ = wrapped.step(state, action, mask, angle, None)
    wrapped.set_residual_context(torch.ones(8))
    adapted_prediction, _ = wrapped.step(state, action, mask, angle, None)
    assert not torch.allclose(adapted_prediction[:, :10], zero_prediction[:, :10])
    torch.testing.assert_close(adapted_prediction[:, 10:], zero_prediction[:, 10:])


def test_locked_residual_ablation_emits_locked_coordinate_correction():
    torch.manual_seed(17)
    adapter = ProjectedResidualInnovation(
        latent_dim=8, rank=8, project_free_coordinates=False)
    state, action, mask, _ = _inputs()
    mask[:, 2] = 1.0
    correction = adapter(state, action, mask, torch.ones(8))
    assert torch.count_nonzero(correction[:, 2]) > 0
    assert torch.count_nonzero(correction[:, 7]) > 0


def test_physical_correction_limits_bound_each_free_coordinate():
    adapter = ProjectedResidualInnovation(
        position_limit=0.0015, velocity_limit=0.025)
    state, action, mask, _ = _inputs()
    correction = adapter(state, action, mask, torch.full((8,), 1e6))
    assert correction[:, :5].abs().max() <= 0.0015 + 1e-7
    assert correction[:, 5:10].abs().max() <= 0.025 + 1e-7


def test_factorized_context_preserves_physical_axes():
    adapter = ProjectedResidualInnovation(
        latent_dim=8, rank=8, factorized_context=True)
    assert not adapter.context_coefficients.weight.requires_grad
    torch.testing.assert_close(
        adapter.context_coefficients.weight, torch.eye(8))


def test_joint_factorized_basis_has_no_topology_input_or_cross_joint_head():
    adapter = ProjectedResidualInnovation(joint_factorized_basis=True)
    assert adapter.transition_basis is None
    assert len(adapter.joint_transition_bases) == 5
    assert all(expert[0].in_features == 15
               and expert[-1].out_features == 2 * adapter.rank
               for expert in adapter.joint_transition_bases)


def test_recurrent_adapter_keeps_k0_exact_and_updates_memory():
    adapter = ProjectedResidualInnovation(memory_dim=12)
    state, action, mask, _ = _inputs()
    correction, memory = adapter.step(state, action, mask, torch.zeros(8), None)
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    assert memory.shape == (state.shape[0], 12)
    assert torch.count_nonzero(memory) > 0


def test_analytic_history_is_fixed_observable_state_and_keeps_k0_exact():
    adapter = ProjectedResidualInnovation(analytic_history=True)
    state, action, mask, _ = _inputs()
    correction, memory = adapter.step(state, action, mask, torch.zeros(8), None)
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    assert memory.shape == (state.shape[0], 10)
    torch.testing.assert_close(memory[:, :5], action)
    torch.testing.assert_close(memory[:, 5:], state[:, 5:10])
    assert adapter.memory_cell is None


def test_shared_joint_basis_has_no_damage_mask_input():
    adapter = ProjectedResidualInnovation(
        analytic_history=True, shared_joint_basis=True)
    assert adapter.transition_basis is None
    assert adapter.joint_transition_bases is None
    # robot(10) + projected action(5) + analytic history(25) + joint id(5)
    assert adapter.shared_joint_transition_basis[0].in_features == 45


def test_before_object_wrapper_preserves_context_rollout_depth():
    base = BlockTriangularDPWM(
        compact_bridge_object_head=True, intervention_object_rank=8,
        intervention_context_dim=8, intervention_context_rank=4,
        intervention_context_ramp=0.1, intervention_context_ramp_start=2,
        intervention_context_delayed=True)
    base.set_intervention_context(torch.ones(8))
    wrapped = FewShotProjectedModel(base, ProjectedResidualInnovation())
    state, action, mask, angle = _inputs(batch=2)
    prediction, hidden = wrapped.step(state, action, mask, angle, None)
    assert isinstance(hidden.base, tuple)
    torch.testing.assert_close(hidden.base[1], torch.ones(2))
    _, hidden = wrapped.step(prediction, action, mask, angle, hidden)
    torch.testing.assert_close(hidden.base[1], torch.full((2,), 2.0))
