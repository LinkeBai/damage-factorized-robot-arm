from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotarm.envs.damage import D2, D3, D4
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.fixed_transform_kinematics import (
    FixedTransformChain,
    rotation_error_degrees,
)


@pytest.fixture(scope="module")
def env() -> MujocoArmEnv:
    return MujocoArmEnv(xml_path="sim/assets/arm_push.xml")


def _mujoco_pose(env: MujocoArmEnv, q: np.ndarray) -> np.ndarray:
    env.data.qpos[env._qpos_adr] = q
    mujoco.mj_forward(env.model, env.data)
    site = env.model.site("ee").id
    pose = np.eye(4)
    pose[:3, :3] = env.data.site_xmat[site].reshape(3, 3)
    pose[:3, 3] = env.data.site_xpos[site]
    return pose


@pytest.mark.parametrize("damage_factory", [D2, D3, D4])
def test_contracted_pose_matches_full_chain_and_mujoco(env, damage_factory) -> None:
    chain = FixedTransformChain()
    damage = damage_factory()
    rng = np.random.default_rng(20260820 + damage.locked[0])
    for _ in range(100):
        q = rng.uniform(env.joint_ranges[:, 0], env.joint_ranges[:, 1])
        q[damage.locked] = damage.lock_angle[damage.locked]
        free = q[damage.joint_mask == 0]
        full = chain.forward_pose(q, damage.joint_mask, damage.lock_angle)
        contracted = chain.contracted_forward_pose(
            free, damage.joint_mask, damage.lock_angle
        )
        expected = _mujoco_pose(env, q)
        assert np.linalg.norm(contracted[:3, 3] - full[:3, 3]) < 1e-12
        assert rotation_error_degrees(contracted, full) < 1e-6
        assert np.linalg.norm(contracted[:3, 3] - expected[:3, 3]) < 1e-9
        assert rotation_error_degrees(contracted, expected) < 1e-6


def test_contracted_edge_retains_locked_rotation() -> None:
    chain = FixedTransformChain()
    damage = D3()
    edges = chain.contract(damage.joint_mask, damage.lock_angle)
    bridge = next(edge for edge in edges if edge.contracted_joints == (2,))
    assert rotation_error_degrees(np.eye(4), bridge.transform) == pytest.approx(
        np.rad2deg(0.5), abs=1e-9
    )
