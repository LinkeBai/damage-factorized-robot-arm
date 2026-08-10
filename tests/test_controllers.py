from __future__ import annotations

import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import (
    JointReferenceConfig,
    jacobian_reach_action,
    joint_reference_action,
    position_jacobian,
    solve_reach_reference,
)
from robotarm.training.sim_protocol import damage_from_name


def test_position_jacobian_shape_and_finiteness():
    jacobian = position_jacobian(np.zeros(5))
    assert jacobian.shape == (3, 5)
    assert np.isfinite(jacobian).all()


def test_healthy_reach_controller_reaches_nominal_target():
    target = np.array([0.252650, 0.265573, 0.140467])
    env = MujocoArmEnv()
    observation = env.reset(target=target)
    for _ in range(250):
        action = jacobian_reach_action(observation["state"], target)
        observation = env.step(action)["observation"]
        if np.linalg.norm(env.ee_pos() - target) < 0.05:
            break
    assert np.linalg.norm(env.ee_pos() - target) < 0.05


def test_locked_joint_action_is_zero():
    action = jacobian_reach_action(
        np.zeros(10), np.array([0.2, 0.1, 0.3]), locked_joints=(1, 2)
    )
    assert action[1] == 0.0
    assert action[2] == 0.0


def test_global_reference_escapes_d3_local_minimum():
    target = np.array([0.102650, -0.309427, 0.065467])
    damage = damage_from_name("D3")
    env = MujocoArmEnv()
    reference, ik_error = solve_reach_reference(
        target,
        env.joint_ranges,
        locked_joints={2: damage.lock_angle_of(2)},
        config=JointReferenceConfig(global_samples=5_000),
    )
    assert ik_error < 0.05
    observation = env.reset(target=target, damage_config=damage)
    for _ in range(200):
        action = joint_reference_action(
            observation["state"], reference, locked_joints=(2,)
        )
        observation = env.step(action)["observation"]
        if np.linalg.norm(env.ee_pos() - target) < 0.05:
            break
    assert np.linalg.norm(env.ee_pos() - target) < 0.05
