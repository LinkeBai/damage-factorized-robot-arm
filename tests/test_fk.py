"""Tests for the analytic FK (tests against the MuJoCo ``ee`` site).

Per PROJECT-PLAN-V4 §9 / G0, the FK endpoint error relative to the simulation
(and later the real arm) must be within task tolerance. Here we compare the
analytic FK against the MuJoCo ``ee`` site position, which is the ground truth
for the simulated chain.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotarm.envs.fk import (
    BASE_HEIGHT,
    L_DISTAL,
    L_UPPER,
    SHOULDER_X,
    J5_TO_J6_Y,
    TCP_OFFSET_X,
    forward_kinematics,
    inverse_kinematics,
)
from robotarm.envs.mujoco_env import MujocoArmEnv


@pytest.fixture(scope="module")
def env():
    return MujocoArmEnv()


def _site_tip(env, q) -> np.ndarray:
    env.data.qpos[:] = np.asarray(q, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    return env.data.site_xpos[env.model.site("ee").id].copy()


def test_zero_pose(env):
    q = np.zeros(5)
    assert np.allclose(_site_tip(env, q), forward_kinematics(q), atol=1e-12)
    # Fully extended arm points straight up; height = base + upper + forearm + tool.
    assert forward_kinematics(q)[0] == pytest.approx(SHOULDER_X + TCP_OFFSET_X)
    assert forward_kinematics(q)[1] == pytest.approx(J5_TO_J6_Y)
    assert forward_kinematics(q)[2] == pytest.approx(BASE_HEIGHT + L_UPPER + L_DISTAL)


def test_pose1_matches_site(env):
    q = np.array([0.5, 0.3, 0.2, 0.1, -0.2])
    assert np.allclose(_site_tip(env, q), forward_kinematics(q), atol=1e-9)


def test_random_poses_within_tolerance(env):
    rng = np.random.default_rng(7)
    # Joint limits from hardware/arm_spec.yaml (SAFE ranges).
    limits = np.array([
        [-1.5708,  1.5708],   # J1: yaw
        [-1.3090,  1.3963],   # J2: shoulder pitch
        [-1.3090,  1.3963],   # J3: elbow pitch
        [-1.3090,  1.4835],   # J4: wrist pitch
        [-1.5708,  1.5708],   # J5: wrist roll
    ])
    worst = 0.0
    for _ in range(500):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        err = np.linalg.norm(_site_tip(env, q) - forward_kinematics(q))
        worst = max(worst, err)
    # FK reproduces the MuJoCo site to machine precision; tolerance reflects
    # the plan's requirement that endpoint error stays within a small margin.
    assert worst < 1e-9


def test_rotational_invariance_under_j1(env):
    # Rotating q1 by 90 deg about the base must rotate the tip about Z.
    q = np.array([0.0, 0.5, -0.3, 0.2, 0.1])
    p0 = forward_kinematics(q)
    q90 = q.copy()
    q90[0] = np.pi / 2
    p90 = forward_kinematics(q90)
    # Same height, and horizontal vector rotated: (x,y) -> (-y,x)
    assert p90[2] == pytest.approx(p0[2], abs=1e-9)
    assert np.allclose(p90[:2], [-p0[1], p0[0]], atol=1e-9)


def test_inverse_kinematics_reaches_nominal_target(env):
    target_q = np.array([0.2, 0.3, -0.25, 0.15, 0.0])
    target = forward_kinematics(target_q)
    q, error = inverse_kinematics(target, env.joint_ranges)
    assert error < 2e-3
    assert np.linalg.norm(forward_kinematics(q) - target) < 2e-3


def test_inverse_kinematics_respects_locked_joint(env):
    target_q = np.array([0.1, 0.4, -0.2, 0.1, 0.0])
    target = forward_kinematics(target_q)
    q, _ = inverse_kinematics(target, env.joint_ranges, locked={1: 0.4})
    assert q[1] == pytest.approx(0.4)
