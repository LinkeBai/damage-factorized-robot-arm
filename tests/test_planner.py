from __future__ import annotations

import numpy as np
import torch

from robotarm.envs.fk import forward_kinematics
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.planner import (
    CEMPlanner,
    PlannerConfig,
    RobustPushCEMPlanner,
    torch_forward_kinematics,
)
from robotarm.models.world_model import WorldModel, WorldModelConfig


def test_torch_fk_matches_numpy_fk():
    q = torch.tensor(
        [[0.2, 0.3, -0.1, 0.15, 0.4]],
        dtype=torch.float32,
    )
    expected = forward_kinematics(q[0].numpy())
    actual = torch_forward_kinematics(q)[0].numpy()
    assert np.allclose(actual, expected, atol=1e-6)


def test_cem_planner_is_frozen_and_masks_locked_joint():
    wm = WorldModel()
    before = [parameter.detach().clone() for parameter in wm.parameters()]
    planner = CEMPlanner(
        wm,
        PlannerConfig(horizon=2, candidates=16, elites=4, iterations=2),
    )
    env = MujocoArmEnv()
    action = planner.plan(
        torch.zeros(10),
        torch.zeros(72),
        torch.tensor([0.2, 0.0, 0.25]),
        torch.as_tensor(env.joint_ranges),
        locked_joints=(2,),
    )
    assert action.shape == (5,)
    assert action[2] == 0.0
    assert torch.all(action.abs() <= 1.0)
    for original, current in zip(before, wm.parameters()):
        assert torch.equal(original, current)


def test_robust_push_planner_masks_locked_joint():
    model = WorldModel(WorldModelConfig(state_dim=14, context_dim=64))
    planner = RobustPushCEMPlanner(
        [model, model], [torch.zeros(64), torch.zeros(64)],
        PlannerConfig(horizon=2, candidates=8, elites=2, iterations=1),
    )
    action = planner.plan(
        torch.zeros(14), torch.tensor([0.28, 0.10]),
        torch.tensor([[-2.0, 2.0]] * 5), locked_joints=(2,),
    )
    assert action.shape == (5,)
    assert action[2] == 0.0
