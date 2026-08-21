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
