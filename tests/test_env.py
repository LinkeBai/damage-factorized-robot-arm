"""Tests for MujocoArmEnv conformance to the RobotEnv protocol + damage."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.damage import DamageConfig
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.protocol import RobotEnv


@pytest.fixture
def env():
    return MujocoArmEnv()


def test_env_satisfies_protocol(env):
    assert isinstance(env, RobotEnv)
    assert env.action_dim == 5
    assert env.observation_dim == 10


def test_reset_and_observe(env):
    obs = env.reset(target=np.array([0.25, 0.1, 0.3]))
    assert set(obs) == {"state", "target"}
    assert obs["state"].shape == (10,)
    assert np.allclose(obs["target"], [0.25, 0.1, 0.3])


def test_step_returns_contract(env):
    obs = env.reset(target=np.array([0.25, 0.1, 0.3]))
    res = env.step(np.zeros(5))
    assert set(res) == {"observation", "reward", "success", "done"}
    assert res["observation"]["state"].shape == (10,)
    assert isinstance(res["reward"], float)
    assert isinstance(res["success"], bool)
    assert isinstance(res["done"], bool)


def test_wrong_action_dim_rejected(env):
    env.reset(target=np.zeros(3))
    with pytest.raises(ValueError):
        env.step(np.zeros(6))


def test_damage_pins_locked_joint(env):
    dmg = DamageConfig.lock_single(2, 0.7)
    env.reset(target=np.zeros(3), damage_config=dmg)
    assert env.data.qpos[2] == pytest.approx(0.7)

    # Run several steps with large commands; locked joint must stay pinned.
    env.step(np.ones(5))
    env.step(np.ones(5))
    assert env.data.qpos[2] == pytest.approx(0.7)
    assert env.data.qvel[2] == pytest.approx(0.0)
    # Unlocked joints are free to move.
    assert env.data.qpos[0] != pytest.approx(0.0, abs=1e-6)


def test_intact_damage_does_not_pin(env):
    env.reset(target=np.zeros(3), damage_config=DamageConfig.intact())
    env.step(np.zeros(5))
    # No joint pinned: qpos still close to zero (small natural motion), not forced.
    assert env.damage_config.n_locked == 0


def test_emergency_stop(env):
    env.reset(target=np.zeros(3))
    env.step(np.ones(5))
    env.emergency_stop()
    assert np.all(env.data.ctrl == 0)


def test_close_runs(env):
    env.close()  # should not raise


def test_mesh_model_uses_same_five_dof_contract():
    mesh = MujocoArmEnv(model_variant="mesh")
    obs = mesh.reset(target=np.array([0.3, 0.1, 0.3]))
    result = mesh.step(np.zeros(5))
    assert mesh.model.nmesh == 7
    assert mesh.action_dim == 5
    assert obs["state"].shape == (10,)
    assert result["observation"]["state"].shape == (10,)
