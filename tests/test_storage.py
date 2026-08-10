"""Tests for append-only storage (PROJECT-PLAN-V4 §10.2)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.data.schema import Episode, StepRecord
from robotarm.data.storage import EpisodeDataset


def make_episode(
    ep_id: str = "ep_a",
    n_steps: int = 3,
    damage_mask: tuple[int, ...] = (0, 0, 1, 0, 0),
) -> Episode:
    steps = []
    for i in range(n_steps):
        state = np.arange(10, dtype=np.float64) + i
        act = np.zeros(5) + 0.1 * i
        steps.append(
            StepRecord(
                observation={"state": state.copy(), "target": np.zeros(3)},
                action_commanded=act.copy(),
                action_applied=act.copy(),
                next_observation={"state": state + 1.0, "target": np.zeros(3)},
                reward=float(i),
                success=i >= n_steps - 1,
                done=i == n_steps - 1,
                safety_flags={"velocity_ok": True, "current_ok": False},
                hardware_state={"temp_c": 30.0 + i, "voltage": 12.0},
            )
        )
    return Episode(
        episode_id=ep_id,
        timestamp_ns=1_700_000_000_000_000_000,
        platform="sim",
        task_id="reach",
        target_id=f"tgt_{ep_id}",
        split="calibration",
        damage_id="D2",
        joint_mask=np.array(damage_mask, dtype=np.int64),
        lock_angle=np.array([0, 0, 0.5, 0, 0], dtype=np.float64),
        steps=steps,
        seed=3,
        config_hash="cfg_1",
        git_commit="deadbeef",
    )


@pytest.fixture
def ds(tmp_path):
    return EpisodeDataset(root=tmp_path / "ds", version="v1")


def write_two(ds):
    ds.add(make_episode("ep_a"))
    ds.add(make_episode("ep_b", damage_mask=(1, 0, 0, 0, 0)))


def test_add_and_len(ds):
    write_two(ds)
    assert len(ds) == 2
    assert ds.ids() == ["ep_a", "ep_b"]


def test_roundtrip_fields_identical(ds):
    ep = make_episode("ep_a", n_steps=4)
    ds.add(ep)
    loaded = ds["ep_a"]
    assert loaded.episode_id == ep.episode_id
    assert loaded.platform == ep.platform
    assert loaded.task_id == ep.task_id
    assert loaded.damage_id == ep.damage_id
    assert loaded.n_transitions == ep.n_transitions
    assert np.array_equal(loaded.joint_mask, ep.joint_mask)
    assert np.allclose(loaded.lock_angle, ep.lock_angle)
    # Per-step fields
    for a, b in zip(loaded.steps, ep.steps):
        assert np.allclose(a.action_commanded, b.action_commanded)
        assert np.allclose(a.observation["state"], b.observation["state"])
        assert np.allclose(a.next_observation["state"], b.next_observation["state"])
        assert a.reward == b.reward
        assert a.success == b.success
        assert a.done == b.done
        assert a.safety_flags == b.safety_flags
        assert a.hardware_state == b.hardware_state


def test_append_only_rejects_duplicate(ds):
    ds.add(make_episode("ep_a"))
    with pytest.raises(ValueError):
        ds.add(make_episode("ep_a"))


def test_reload_from_disk_persists(tmp_path):
    root = tmp_path / "ds"
    ds1 = EpisodeDataset(root=root, version="v1")
    ds1.add(make_episode("ep_a"))
    ds1.add(make_episode("ep_b"))

    ds2 = EpisodeDataset(root=root, version="v1")
    assert len(ds2) == 2
    assert ds2["ep_b"].target_id == "tgt_ep_b"


def test_exclude_marks_not_deletes(ds):
    ds.add(make_episode("ep_a"))
    path = ds._payload_path("ep_a")
    ds.exclude("ep_a", "bad reward")
    assert ds._entries["ep_a"].excluded
    assert ds._entries["ep_a"].exclusion_reason == "bad reward"
    assert path.exists()  # payload untouched
    # Excluded is still loadable/iterated in manifest but flagged.
    assert ds["ep_a"].episode_id == "ep_a"


def test_clean_version_skips_excluded(tmp_path):
    root = tmp_path / "ds"
    ds = EpisodeDataset(root=root, version="v1")
    ds.add(make_episode("ep_a"))
    ds.add(make_episode("ep_b"))
    ds.exclude("ep_b", "keep out")

    new_root = tmp_path / "ds_clean"
    clean = ds.clean_version(new_root=new_root, new_version="v2")
    assert len(clean) == 1
    assert clean.ids() == ["ep_a"]
    assert clean.version == "v2"
    assert (new_root / "manifest.json").exists()


def test_clean_version_refuses_existing(tmp_path):
    root = tmp_path / "ds"
    ds = EpisodeDataset(root=root)
    ds.add(make_episode("ep_a"))
    with pytest.raises(FileExistsError):
        ds.clean_version(new_root=root, new_version="v2")  # same dir exists


def test_verify_integrity_passes(ds):
    ds.add(make_episode("ep_a"))
    assert all(ds.verify_integrity().values())


def test_integrity_detects_corruption(ds):
    ds.add(make_episode("ep_a"))
    path = ds._payload_path("ep_a")
    path.write_bytes(b"corrupted!")
    # Fresh load of manifest still fine because manifest is separate.
    assert ds.verify_integrity()["ep_a"] is False


def test_sweep_invalid_marks_bad(tmp_path):
    root = tmp_path / "ds"
    ds = EpisodeDataset(root=root)
    ds.add(make_episode("good"))
    # Corrupt the "bad" episode's payload on disk so it fails to load/validate.
    ds.add(make_episode("bad"))
    ds._payload_path("bad").write_bytes(b"garbage-not-a-npz")

    cleaned = ds.sweep_invalid()
    assert cleaned == ["bad"]
    assert ds._entries["bad"].excluded
    assert not ds._entries["good"].excluded
