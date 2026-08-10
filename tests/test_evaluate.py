"""Tests for the evaluation harness (PROJECT-PLAN-V4 §5.1 metrics)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.damage import DamageConfig
from robotarm.training.evaluate import (
    ReachMetrics,
    _rand_policy,
    _zero_policy,
    evaluate_reach,
)


@pytest.fixture(scope="module")
def env():
    return MujocoArmEnv()


def _near_targets():
    # Targets close to the rest pose so a zero policy can't accidentally succeed
    # but still probes the physics/dynamics.
    return np.array([[0.25, 0.1, 0.2], [0.2, -0.05, 0.25]])


def test_zero_policy_metrics_well_formed(env):
    t = _near_targets()
    m = evaluate_reach(env, t, _zero_policy, max_steps=30)
    assert isinstance(m, ReachMetrics)
    assert 0.0 <= m.success_rate <= 1.0
    assert m.mean_final_distance >= 0.0
    assert m.mean_time_to_reach > 0


def test_metrics_differ_by_policy(env):
    t = _near_targets()
    mz = evaluate_reach(env, t, _zero_policy, max_steps=40, rng=np.random.default_rng(0))
    mr = evaluate_reach(env, t, _rand_policy(np.random.default_rng(0)), max_steps=40)
    print("zero", mz.as_dict(), "rand", mr.as_dict())
    # Both should at least run without error; success rates are float.
    assert isinstance(mz.success_rate, float)
    assert isinstance(mr.success_rate, float)


def test_damage_config_is_applied(env):
    t = _near_targets()
    dmg = DamageConfig.lock_single(2, 0.5)
    m = evaluate_reach(env, t[:1], _zero_policy, max_steps=20, damage_config=dmg)
    # After reset under damage, joint 2 pinned.
    obs = env.reset(target=t[0], damage_config=dmg)
    assert env.data.qpos[2] == pytest.approx(0.5)


def test_max_steps_bounds_time(env):
    t = _near_targets()[:1]
    m = evaluate_reach(env, t, _zero_policy, max_steps=15)
    assert m.mean_time_to_reach <= 15