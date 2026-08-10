from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.residual_physics import (
    RESIDUAL_PROFILES,
    ResidualPhysicsConfig,
    residual_profile,
)
from robotarm.training.sim_data import collect_trajectory
from robotarm.training.sim_protocol import DomainSpec, build_g1_protocol


def test_residual_config_validation():
    with pytest.raises(ValueError):
        ResidualPhysicsConfig(actuator_scale=(1.0,))
    with pytest.raises(ValueError):
        ResidualPhysicsConfig(control_delay_steps=-1)
    assert residual_profile("mixed_unseen").control_delay_steps == 2
    assert set(RESIDUAL_PROFILES) >= {"nominal", "weak_motor", "mixed_unseen"}


def test_delay_and_deadband_change_applied_action():
    config = ResidualPhysicsConfig(
        control_delay_steps=1,
        action_deadband=0.1,
    )
    env = MujocoArmEnv(residual_physics=config)
    env.reset(target=np.zeros(3))
    env.step(np.array([0.5, 0.05, 0.0, 0.0, 0.0]))
    assert np.allclose(env.last_applied_action, 0.0)
    env.step(np.zeros(5))
    assert np.allclose(env.last_applied_action, [0.5, 0.0, 0.0, 0.0, 0.0])


def test_residual_scales_model_and_loads_payload():
    nominal = MujocoArmEnv()
    config = ResidualPhysicsConfig(
        damping_scale=2.0,
        armature_scale=1.5,
        payload_mass_delta_kg=0.02,
    )
    changed = MujocoArmEnv(residual_physics=config)
    assert np.allclose(
        changed.model.dof_damping[changed._qvel_adr],
        nominal.model.dof_damping[nominal._qvel_adr] * 2.0,
    )
    assert changed.model.body("tool").mass > nominal.model.body("tool").mass


def test_protocol_holds_out_combinations():
    protocol = build_g1_protocol()
    train = {domain.domain_id for domain in protocol.train}
    validation = {domain.domain_id for domain in protocol.validation}
    test = {domain.domain_id for domain in protocol.test}
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert {domain.topology for domain in protocol.test} == {
        "D2",
        "D3",
    }
    assert {domain.residual_name for domain in protocol.test} == {"mixed_composition"}
    assert protocol.dof == 5
    assert len(protocol.sha256) == 64
    assert protocol.calibration_shots == (0, 1, 2, 5)


def test_collect_trajectory_records_commanded_and_applied():
    domain = DomainSpec("D2", "delay_1", "test")
    trajectory = collect_trajectory(domain, steps=6, seed=4)
    assert trajectory.states.shape == (7, 10)
    assert trajectory.actions.shape == (6, 5)
    assert trajectory.applied_actions.shape == (6, 5)
    assert np.allclose(trajectory.actions[:, 1], 0.0)
    assert not np.allclose(trajectory.actions.numpy(), trajectory.applied_actions.numpy())
