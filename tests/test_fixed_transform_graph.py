from __future__ import annotations

import mujoco
import torch

from robotarm.models.fixed_transform_graph import (
    FixedTransformContactWorldModel,
    FixedTransformGraphObjectWorldModel,
    FixedTransformGraphWorldModel,
)


def test_joint_poses_change_downstream_with_locked_rotation() -> None:
    model = FixedTransformGraphWorldModel()
    q = torch.zeros(1, 5)
    base_positions, _ = model._joint_poses(q)
    q[:, 2] = -0.5
    locked_positions, _ = model._joint_poses(q)
    assert torch.allclose(base_positions[:, :3], locked_positions[:, :3])
    assert not torch.allclose(base_positions[:, 3:], locked_positions[:, 3:])


def test_fixed_transform_graph_enforces_lock_and_backpropagates() -> None:
    model = FixedTransformGraphWorldModel()
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert prediction.shape == state.shape
    assert hidden.shape == (2, 5, 128)
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))
    prediction[:, :10].pow(2).mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_object_loss_is_isolated_from_joint_transition() -> None:
    model = FixedTransformGraphObjectWorldModel()
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction[:, 10:].pow(2).mean().backward()
    object_parameters = set(model.object_head.parameters())
    assert all(parameter.grad is not None for parameter in object_parameters)
    joint_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter not in object_parameters and parameter.grad is not None
    ]
    assert all(torch.count_nonzero(gradient) == 0 for gradient in joint_gradients)
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))


def test_contact_world_model_has_geometric_gate_and_exact_lock() -> None:
    model = FixedTransformContactWorldModel()
    state, action = torch.zeros(2, 14), torch.zeros(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    mask[:, 2], angle[:, 2] = 1.0, -0.5
    state[:, 2] = -0.5
    endpoints = model._pusher_endpoints_xy(state[:, :5])
    state[0, 10:12] = endpoints[0, 0]
    state[1, 10:12] = torch.tensor([0.45, 0.45])
    features, gate = model._contact_features(state, state[:, :5])
    prediction, hidden = model.step(state, action, mask, angle, None)
    assert features.shape == (2, 20)
    assert gate[0] > gate[1]
    assert prediction.shape == (2, 14)
    assert hidden.shape == (2, 5, 128)
    assert torch.allclose(prediction[:, 2], torch.full((2,), -0.5))
    assert torch.allclose(prediction[:, 7], torch.zeros(2))


def test_contact_pusher_start_matches_mujoco_ee() -> None:
    from robotarm.envs.mujoco_env import MujocoArmEnv

    model = FixedTransformContactWorldModel()
    env = MujocoArmEnv(xml_path="sim/assets/arm_push.xml")
    q = torch.tensor([[0.2, -0.4, 0.3, -0.1, 0.5]])
    env.data.qpos[env._qpos_adr] = q[0].numpy()
    mujoco.mj_forward(env.model, env.data)
    expected = torch.as_tensor(env.data.site("ee").xpos[:2].copy())
    actual = model._pusher_endpoints_xy(q)[0, 0].double()
    assert torch.allclose(actual, expected, atol=1e-7)


def test_contact_object_loss_does_not_update_joint_path() -> None:
    model = FixedTransformContactWorldModel()
    state, action = torch.randn(2, 14), torch.randn(2, 5)
    mask, angle = torch.zeros(2, 5), torch.zeros(2, 5)
    prediction, _ = model.step(state, action, mask, angle, None)
    prediction[:, 10:].pow(2).mean().backward()
    object_modules = (
        model.object_contact_encoder, model.object_free_head, model.object_impulse_head
    )
    object_parameters = {
        parameter for module in object_modules for parameter in module.parameters()
    }
    assert all(parameter.grad is not None for parameter in object_parameters)
    joint_gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter not in object_parameters and parameter.grad is not None
    ]
    assert all(torch.count_nonzero(gradient) == 0 for gradient in joint_gradients)
