import torch

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM


def _inputs(batch=3):
    return (torch.randn(batch, 14), torch.randn(batch, 5),
            torch.zeros(batch, 5), torch.zeros(batch, 5))


def test_bt_dpwm_step_shape_and_damage_projection():
    model = BlockTriangularDPWM()
    state, action, mask, angle = _inputs()
    mask[:, 2], angle[:, 2] = 1.0, 0.37
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == state.shape
    assert hidden.shape == (3, 5, model.cfg.hidden_dim)
    torch.testing.assert_close(prediction[:, 2], angle[:, 2])
    torch.testing.assert_close(prediction[:, 7], torch.zeros(3))


def test_object_loss_cannot_update_robot_block():
    model = BlockTriangularDPWM()
    prediction, _ = model.step(*_inputs(), None)
    prediction[:, 10:].pow(2).mean().backward()
    robot_parameters = [parameter for name, parameter in model.named_parameters()
                        if name.startswith("robot_")]
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in robot_parameters)
    assert any(p.grad is not None for p in model.object_head.parameters())


def test_joint_prediction_is_invariant_to_object_state():
    model = BlockTriangularDPWM().eval()
    state, action, mask, angle = _inputs()
    changed = state.clone(); changed[:, 10:] += 100.0
    first, _ = model.step(state, action, mask, angle, None)
    second, _ = model.step(changed, action, mask, angle, None)
    torch.testing.assert_close(first[:, :10], second[:, :10])


def test_contact_conditioning_changes_joint_forward_but_preserves_gradient_boundary():
    model = BlockTriangularDPWM(contact_conditioned_robot=True)
    state, action, mask, angle = _inputs()
    changed = state.clone(); changed[:, 10:] += 100.0
    first, _ = model.step(state, action, mask, angle, None)
    second, _ = model.step(changed, action, mask, angle, None)
    assert not torch.allclose(first[:, :10], second[:, :10])
    first[:, 10:].pow(2).mean().backward()
    robot_parameters = [parameter for name, parameter in model.named_parameters()
                        if name.startswith("robot_")]
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in robot_parameters)


def test_independent_object_encoder_has_own_recurrent_representation():
    model = BlockTriangularDPWM(
        contact_conditioned_robot=True, independent_object_encoder=True
    )
    prediction, hidden = model.step(*_inputs(), None)
    assert hidden.shape == (3, 10, model.cfg.hidden_dim)
    prediction[:, 10:].pow(2).mean().backward()
    robot_parameters = [parameter for name, parameter in model.named_parameters()
                        if name.startswith("robot_")]
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in robot_parameters)
    assert any(p.grad is not None for p in model.object_encoder.parameters())


def test_asymmetric_hidden_widths_preserve_directed_gradient_boundary():
    model = BlockTriangularDPWM(
        contact_conditioned_robot=True,
        independent_object_encoder=True,
        object_hidden_dim=48,
    )
    prediction, hidden = model.step(*_inputs(), None)
    assert isinstance(hidden, tuple)
    assert hidden[0].shape == (3, 5, 96)
    assert hidden[1].shape == (3, 5, 48)
    prediction[:, 10:].pow(2).mean().backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameter in model.named_parameters() if name.startswith("robot_")
    )


def test_zero_initialized_reaction_adapter_preserves_forward_then_receives_joint_gradient():
    torch.manual_seed(3)
    plain = BlockTriangularDPWM(contact_conditioned_robot=True)
    torch.manual_seed(3)
    adapted = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8)
    adapted.load_state_dict({**adapted.state_dict(), **plain.state_dict()})
    inputs = _inputs()
    first, _ = plain.step(*inputs, None)
    second, _ = adapted.step(*inputs, None)
    torch.testing.assert_close(first, second)
    second[:, :10].pow(2).mean().backward()
    assert any(parameter.grad is not None for parameter in adapted.reaction_adapter.parameters())


def test_geometry_gated_reaction_adds_no_parameters_and_suppresses_far_object():
    plain = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8)
    gated = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8,
                               reaction_geometry_gate=True)
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in gated.parameters())
    q = torch.zeros(2, 5)
    near = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    gate = gated._reaction_contact_gate(q, near)
    assert gate[0] > gate[1]
    assert gate[1] < 1e-4


def test_zero_reaction_scale_recovers_unadapted_forward():
    torch.manual_seed(13)
    plain = BlockTriangularDPWM(contact_conditioned_robot=True)
    torch.manual_seed(13)
    scaled = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8,
                                reaction_scale=0.0)
    scaled.load_state_dict({**scaled.state_dict(), **plain.state_dict()})
    inputs = _inputs()
    first, _ = plain.step(*inputs, None)
    second, _ = scaled.step(*inputs, None)
    torch.testing.assert_close(first, second)


def test_physical_reaction_adapter_is_hidden_basis_independent_and_budget_smaller():
    latent = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8)
    physical = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8,
                                  reaction_physical_features=True)
    assert physical.reaction_adapter[0].in_features == 10
    assert sum(p.numel() for p in physical.parameters()) < sum(p.numel() for p in latent.parameters())


def test_event_trace_is_parameter_free_and_decays_after_contact():
    model = BlockTriangularDPWM(contact_conditioned_robot=True,
                               independent_object_encoder=True,
                               object_hidden_dim=32, reaction_rank=8,
                               reaction_geometry_gate=True,
                               reaction_event_decay=0.5)
    before = sum(p.numel() for p in model.parameters())
    inputs = _inputs(batch=2)
    _, hidden = model.step(*inputs, None)
    assert isinstance(hidden, tuple) and len(hidden) == 3
    assert hidden[2].shape == (2,)
    assert sum(p.numel() for p in model.parameters()) == before


def test_fixed_reaction_initialization_is_seed_invariant():
    torch.manual_seed(1)
    first = BlockTriangularDPWM(reaction_rank=16, reaction_physical_features=True,
                               reaction_fixed_initialization=True)
    torch.manual_seed(999)
    second = BlockTriangularDPWM(reaction_rank=16, reaction_physical_features=True,
                                reaction_fixed_initialization=True)
    torch.testing.assert_close(first.reaction_adapter[0].weight,
                               second.reaction_adapter[0].weight)
    torch.testing.assert_close(first.reaction_adapter[0].bias,
                               second.reaction_adapter[0].bias)


def test_kinematic_projection_enforces_semi_implicit_position_step():
    model = BlockTriangularDPWM(kinematic_integration_dt=0.005,
                               kinematic_position_blend=1.0)
    state, action, mask, angle = _inputs()
    prediction, _ = model.step(state, action, mask, angle, None)
    torch.testing.assert_close(prediction[:, :5],
                               state[:, :5] + 0.005 * prediction[:, 5:10])


def test_shadow_object_context_fits_budget_and_has_independent_hidden_path():
    from robotarm.models.topology_graph_world_model import TopologyGraphConfig
    model = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=136),
                               contact_conditioned_robot=True,
                               independent_object_encoder=True,
                               object_hidden_dim=32, shadow_object_rank=8)
    prediction, hidden = model.step(*_inputs(), None)
    assert prediction.shape[-1] == 14
    assert isinstance(hidden, tuple) and len(hidden) == 3
    assert hidden[2].shape == (3, 4)
    assert sum(p.numel() for p in model.parameters()) == 338074


def test_two_robot_experts_fit_budget_and_keep_separate_hidden_states():
    from robotarm.models.topology_graph_world_model import TopologyGraphConfig
    model = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=96),
                               contact_conditioned_robot=True,
                               independent_object_encoder=True,
                               object_hidden_dim=32, robot_expert_count=2)
    prediction, hidden = model.step(*_inputs(), None)
    assert prediction.shape == (3, 14)
    assert hidden[0].shape == (3, 10, 96)
    assert hidden[1].shape == (3, 5, 32)
    assert sum(p.numel() for p in model.parameters()) == 337448
