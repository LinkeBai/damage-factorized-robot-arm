import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from robotarm.envs.damage import DamageConfig
from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv


def test_original_arm_push_adapter_has_nontrivial_goal_and_expected_contract():
    env = OriginalArmPushEnv(seed=7, max_episode_steps=3)
    obs = env.reset(seed=7)
    assert obs.shape == (33,)
    assert env.action_space.shape == (5,)
    assert np.linalg.norm(env._target_xy - env._block_initial_xy) >= 0.025
    next_obs, reward, done, info = env.step(np.zeros(5))
    assert next_obs.shape == (33,)
    assert np.isfinite(reward)
    assert isinstance(done, bool)
    assert info["goal_distance_m"] > env.success_tolerance_m


def test_original_arm_push_adapter_preserves_exact_hard_lock():
    damage = DamageConfig.lock_single(2, -0.5)
    env = OriginalArmPushEnv(damage=damage, seed=11)
    obs = env.reset(seed=11)
    assert obs[-10:-5].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    for _ in range(5):
        env.step(np.ones(5))
        assert env._env.joint_positions[2] == pytest.approx(-0.5, abs=1e-12)
        qvel_address = env._env._qvel_adr[2]
        assert env._env.data.qvel[qvel_address] == pytest.approx(0.0, abs=1e-12)


def test_directional_seed_policy_reaches_contact_and_moves_block():
    env = OriginalArmPushEnv(seed=3, seed_policy="directional")
    try:
        env.reset(seed=3)
        had_contact = False
        max_displacement = 0.0
        for _ in range(env.max_episode_steps):
            _, _, done, info = env.step(env.rand_act())
            had_contact |= info["contact"]
            max_displacement = max(max_displacement, info["block_displacement_m"])
            if done:
                break
        assert had_contact
        assert max_displacement > 0.005
    finally:
        env.close()
