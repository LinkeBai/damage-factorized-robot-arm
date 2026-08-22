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


def test_contact_gated_context_is_parameter_free_and_ignores_far_object():
    plain = BlockTriangularDPWM(contact_conditioned_robot=True)
    gated = BlockTriangularDPWM(contact_conditioned_robot=True,
                               contact_gated_object_context=True)
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in gated.parameters())
    state, action, mask, angle = _inputs(batch=2)
    state[:, :10] = 0.0
    action[:] = 0.0
    state[0, 10:12] = torch.tensor([10.0, 10.0])
    state[1, 10:12] = torch.tensor([20.0, 20.0])
    state[:, 12:] = 0.0
    robot, _, _, _, _ = gated.step_robot(state, action, mask, angle, None)
    torch.testing.assert_close(robot[0], robot[1])


def test_gated_shared_scaffold_copy_preserves_one_step_robot_function():
    from robotarm.models.topology_graph_world_model import (
        TopologyGraphConfig, TopologyGraphWorldModel,
    )
    cfg = TopologyGraphConfig(hidden_dim=16, contact_gated_object_context=True)
    shared = TopologyGraphWorldModel(cfg)
    candidate = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=16), contact_conditioned_robot=True,
        contact_gated_object_context=True,
    )
    source, target = shared.state_dict(), candidate.state_dict()
    for left, right in (("node_encoder.", "robot_encoder."),
                        ("message.", "robot_message."),
                        ("update.", "robot_update."),
                        ("temporal.", "robot_temporal."),
                        ("joint_head.", "robot_head.")):
        for name, value in source.items():
            if name.startswith(left):
                target[right + name[len(left):]] = value
    candidate.load_state_dict(target)
    state, action, mask, angle = _inputs()
    shared_prediction, shared_hidden = shared.step(state, action, mask, angle, None)
    robot, candidate_hidden, _, _, _ = candidate.step_robot(
        state, action, mask, angle, None
    )
    torch.testing.assert_close(robot, shared_prediction[:, :10])
    torch.testing.assert_close(candidate_hidden, shared_hidden)


def test_semi_implicit_shared_scaffold_enforces_position_update():
    from robotarm.models.topology_graph_world_model import (
        TopologyGraphConfig, TopologyGraphWorldModel,
    )
    model = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=16, kinematic_integration_dt=0.005,
        kinematic_position_blend=1.0,
    ))
    state, action, mask, angle = _inputs()
    prediction, _ = model.step(state, action, mask, angle, None)
    torch.testing.assert_close(
        prediction[:, :5], state[:, :5] + 0.005 * prediction[:, 5:10]
    )


def test_zero_linear_physical_reaction_preserves_forward_with_22_parameters():
    torch.manual_seed(31)
    plain = BlockTriangularDPWM(contact_conditioned_robot=True)
    torch.manual_seed(31)
    linear = BlockTriangularDPWM(contact_conditioned_robot=True,
                                linear_physical_reaction=True)
    linear.load_state_dict({**linear.state_dict(), **plain.state_dict()})
    inputs = _inputs()
    first, _ = plain.step(*inputs, None)
    second, _ = linear.step(*inputs, None)
    torch.testing.assert_close(first, second)
    assert (sum(p.numel() for p in linear.parameters())
            - sum(p.numel() for p in plain.parameters())) == 22


def test_relative_reaction_clip_has_zero_path_and_bounds_each_joint():
    torch.manual_seed(41)
    plain = BlockTriangularDPWM(contact_conditioned_robot=True)
    torch.manual_seed(41)
    clipped = BlockTriangularDPWM(contact_conditioned_robot=True, reaction_rank=8,
                                 reaction_relative_clip=0.1)
    clipped.load_state_dict({**clipped.state_dict(), **plain.state_dict()})
    with torch.no_grad():
        clipped.reaction_adapter[-1].weight.fill_(1.0)
        clipped.reaction_adapter[-1].bias.fill_(0.5)
    inputs = _inputs()
    base, _, _, _, _ = plain.step_robot(*inputs, None)
    corrected, _, _, _, _ = clipped.step_robot(*inputs, None)
    state = inputs[0]
    base_delta = torch.stack((base[:, :5] - state[:, :5],
                              base[:, 5:10] - state[:, 5:10]), dim=-1)
    correction = torch.stack((corrected[:, :5] - base[:, :5],
                              corrected[:, 5:10] - base[:, 5:10]), dim=-1)
    assert torch.all(torch.linalg.vector_norm(correction, dim=-1)
                     <= 0.10001 * torch.linalg.vector_norm(base_delta, dim=-1) + 1e-6)
    clipped.reaction_relative_clip = 0.0
    recovered, _, _, _, _ = clipped.step_robot(*inputs, None)
    torch.testing.assert_close(recovered, base)


def test_compact_bridge_matches_h136_budget_and_blocks_object_gradient():
    from robotarm.models.topology_graph_world_model import TopologyGraphConfig
    model = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=136), contact_conditioned_robot=True,
        compact_bridge_object_head=True,
    )
    assert sum(p.numel() for p in model.parameters()) == 338102
    prediction, _ = model.step(*_inputs(), None)
    prediction[:, 10:].pow(2).mean().backward()
    robot = [parameter for name, parameter in model.named_parameters()
             if name.startswith("robot_")]
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in robot)
    assert any(p.grad is not None for p in model.object_head.parameters())
