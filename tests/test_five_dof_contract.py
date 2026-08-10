from __future__ import annotations

import numpy as np
import yaml

from robotarm.data.schema import N_JOINTS
from robotarm.envs.damage import D3
from robotarm.envs.mujoco_env import (
    CONTROLLED_ACTUATORS,
    CONTROLLED_JOINTS,
    MujocoArmEnv,
)
from robotarm.envs.residual_physics import ResidualPhysicsConfig
from robotarm.training.target_split import load_target_split


def test_five_joint_names_and_model_contract():
    assert N_JOINTS == 5
    assert CONTROLLED_JOINTS == ("j1", "j2", "j3", "j4", "j5")
    assert CONTROLLED_ACTUATORS == ("m1", "m2", "m3", "m4", "m5")
    for variant in ("simple", "mesh"):
        env = MujocoArmEnv(model_variant=variant)
        assert env.model.nq == 5
        assert env.model.nu == 5
        assert env.action_dim == 5
        assert env.observation_dim == 10


def test_joint_map_matches_action_order():
    with open("hardware/joint_map.yaml", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    joints = mapping["joints"]
    assert [item["index"] for item in joints] == list(range(5))
    assert [item["mujoco"] for item in joints] == list(CONTROLLED_JOINTS)
    assert [item["urdf"] for item in joints] == [
        "Rotation",
        "Rotation1",
        "Rotation2",
        "Rotation3",
        "Rotation4",
    ]


def test_1000_step_smoke_has_no_nan_and_preserves_lock():
    env = MujocoArmEnv()
    env.reset(target=np.array([0.2, 0.0, 0.25]), damage_config=D3())
    rng = np.random.default_rng(4)
    for _ in range(1000):
        action = rng.uniform(-0.2, 0.2, size=5)
        action[2] = 0.0
        result = env.step(action)
        assert np.isfinite(result["observation"]["state"]).all()
    assert env.joint_positions[2] == D3().lock_angle_of(2)


def test_seeded_noisy_rollout_is_reproducible():
    config = ResidualPhysicsConfig(
        observation_noise_std=0.002,
        control_delay_steps=1,
        seed=91,
    )
    env_a = MujocoArmEnv(residual_physics=config)
    env_b = MujocoArmEnv(residual_physics=config)
    target = np.array([0.2, 0.0, 0.25])
    observations_a = [env_a.reset(target=target)["state"]]
    observations_b = [env_b.reset(target=target)["state"]]
    rng = np.random.default_rng(8)
    actions = [rng.uniform(-0.2, 0.2, size=5) for _ in range(20)]
    for action in actions:
        observations_a.append(env_a.step(action)["observation"]["state"])
        observations_b.append(env_b.step(action)["observation"]["state"])
    assert np.array_equal(np.stack(observations_a), np.stack(observations_b))


def test_frozen_target_split_is_disjoint_and_hashed():
    split = load_target_split()
    calibration_ids = {target.target_id for target in split.calibration}
    evaluation_ids = {target.target_id for target in split.evaluation}
    assert not calibration_ids & evaluation_ids
    assert len(split.sha256) == 64
    assert split.status == "frozen_after_g0_2026-08-10"
