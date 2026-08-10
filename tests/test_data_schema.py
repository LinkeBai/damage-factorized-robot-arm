"""Tests for src/robotarm/data/schema.py (§10.1 trajectory schema)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.data.schema import N_JOINTS, Episode, StepRecord


def make_step(reward: float = 0.0, done: bool = False) -> StepRecord:
    action = np.zeros(N_JOINTS)
    state = np.zeros(2 * N_JOINTS)
    return StepRecord(
        observation={"state": state.copy(), "target": np.zeros(3)},
        action_commanded=action.copy(),
        action_applied=action.copy(),
        next_observation={"state": state.copy(), "target": np.zeros(3)},
        reward=reward,
        success=False,
        done=done,
    )


def make_episode() -> Episode:
    a = np.zeros(N_JOINTS, dtype=np.int64)
    b = np.zeros(N_JOINTS)
    return Episode(
        episode_id="ep_001",
        timestamp_ns=1_721_000_000_000_000_000,
        platform="sim",
        task_id="reach",
        target_id="tgt_A1",
        split="calibration",
        damage_id="D0",
        joint_mask=a,
        lock_angle=b,
        steps=[make_step(done=True)],
        seed=0,
        git_commit="abc123",
    )


def test_episode_validate_ok():
    make_episode().validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform", "real_time"),
        ("split", "train"),
        ("episode_id", ""),
    ],
)
def test_episode_validate_rejects_bad_scalars(field, value):
    ep = make_episode()
    setattr(ep, field, value)
    with pytest.raises(ValueError):
        ep.validate()


def test_episode_validate_rejects_empty_steps():
    ep = make_episode()
    ep.steps = []
    with pytest.raises(ValueError):
        ep.validate()


def test_episode_rejects_bad_joint_mask():
    ep = make_episode()
    ep.joint_mask = np.array([0, 1, 1, 1, 1])  # still fine
    ep.validate()
    ep.joint_mask = np.array([2, 0, 0, 0, 0])  # 2 invalid
    with pytest.raises(ValueError):
        ep.validate()


def test_episode_rejects_wrong_action_dim():
    ep = make_episode()
    ep.steps[0].action_commanded = np.zeros(N_JOINTS - 1)
    with pytest.raises(ValueError):
        ep.validate()


def test_missing_state_key_rejected():
    ep = make_episode()
    ep.steps[0].observation = {"target": np.zeros(3)}
    with pytest.raises(ValueError):
        ep.validate()


def test_inconsistent_observation_keys_rejected():
    ep = make_episode()
    ep.steps.append(make_step(done=True))
    ep.steps[1].observation = {"state": np.zeros(2 * N_JOINTS), "target": np.zeros(3), "image": np.zeros((1, 1, 3))}
    with pytest.raises(ValueError):
        ep.validate()


def test_transition_count():
    ep = make_episode()
    ep.steps = [make_step() for _ in range(10)]
    assert ep.length == 10
    assert ep.n_transitions == 9


def test_to_tree_shape():
    tree = make_episode().to_tree()
    assert set(tree) == {
        "episode_id", "timestamp_ns", "platform", "task_id", "target_id",
        "split", "damage_id", "joint_mask", "lock_angle", "seed",
        "config_hash", "git_commit", "camera_frame_ref", "steps",
    }
    assert len(tree["steps"]) == 1
