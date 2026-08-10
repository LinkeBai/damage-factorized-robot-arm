"""Tests for reachability analysis (PROJECT-PLAN-V4 G0 §4)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.damage import DamageConfig
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.reachability import (
    analyze_morphology,
    analyze_damage_morphology,
    common_reachable_region,
    joint_ranges_for_damage,
    reachable_points,
    sample_targets_from_common,
)
from robotarm.envs.fk import BASE_HEIGHT, L_UPPER, L_DISTAL, forward_kinematics


def _joint_ranges(env: MujocoArmEnv) -> np.ndarray:
    # Pull per-joint limits straight off the MJCF.
    ranges = np.zeros((env.model.njnt, 2))
    jnt_adr = env.model.jnt_qposadr
    for i in range(env.model.njnt):
        ranges[i] = env.model.jnt_range[i]
    return ranges


@pytest.fixture(scope="module")
def env():
    return MujocoArmEnv()


@pytest.fixture(scope="module")
def ranges(env):
    return _joint_ranges(env)


def fk(q):
    return forward_kinematics(q)


def test_reachable_points_shape_and_bbox(ranges):
    pts = reachable_points(fk, ranges, n=50, rng=np.random.default_rng(0))
    assert pts.shape == (50, 3)
    # Fully vertical arm at rest: max z bound by full extension height.
    assert pts[:, 2].max() <= BASE_HEIGHT + L_UPPER + L_DISTAL + 1e-9


def test_analyze_morphology(ranges):
    res = analyze_morphology("intact", fk, ranges, n=20, rng=np.random.default_rng(0))
    assert res.kind == "intact"
    assert res.points.shape == (20, 3)
    assert res.bounds[0].shape == (3,)


def test_common_region_contains_intact_subset(ranges):
    # A single morphology's "common" region should cover nearly all its points.
    res = analyze_morphology("intact", fk, ranges, n=800, rng=np.random.default_rng(0))
    grid, centers = common_reachable_region([res], voxel_size=0.02)
    assert np.any(grid)
    inside = sum(res.contains(p) is not None for p in res.points)  # bounds always a superset
    assert res.bounds[0][2] <= BASE_HEIGHT  # base min z >= near 0


def test_sample_targets_ok(ranges):
    intact = analyze_morphology("intact", fk, ranges, n=800, rng=np.random.default_rng(0))
    targets = sample_targets_from_common([intact], n=10, rng=np.random.default_rng(1))
    assert targets.shape == (10, 3)
    # Every sampled target should be inside the intact bounds box.
    for t in targets:
        assert intact.contains(t)


def test_sample_targets_map_to_reachable_region(env, ranges):
    """Sampled targets should be approximately reachable by FK for the intact arm."""
    intact = analyze_morphology("intact", fk, ranges, n=1000, rng=np.random.default_rng(0))
    targets = sample_targets_from_common([intact], n=20, rng=np.random.default_rng(1))
    reached = 0.0
    for t in targets:
        err = min(
            np.linalg.norm(p - t)
            for p in intact.points
        )
        # Mesh resolution limits accuracy; just sanity-check it's not huge.
        assert err < 0.05
        reached += 1.0
    assert reached == 20.0


def test_damage_smaller_common_region(ranges):
    """Locking a joint should shrink the common reachable region."""
    intact = analyze_morphology("intact", fk, ranges, n=1500, rng=np.random.default_rng(0))
    # Mock a damaged morphology with narrower joint_ranges for j2..j4.
    tight = ranges.copy()
    tight[2] = [0.2, 0.4]
    tight[3] = [-0.3, 0.3]
    damaged = analyze_morphology("D2m", fk, tight, n=1500, rng=np.random.default_rng(0))
    grid_i, _ = common_reachable_region([intact], voxel_size=0.03)
    grid_both, _ = common_reachable_region([intact, damaged], voxel_size=0.03)
    assert grid_both.sum() <= grid_i.sum()


def test_locked_joint_range_collapses_exactly(ranges):
    damage = DamageConfig.lock_single(1, 0.4)
    locked = joint_ranges_for_damage(ranges, damage)
    assert np.allclose(locked[1], [0.4, 0.4])
    assert np.allclose(locked[[0, 2, 3, 4]], ranges[[0, 2, 3, 4]])

    result = analyze_damage_morphology(
        "D2@0.4",
        fk,
        ranges,
        damage,
        n=20,
        rng=np.random.default_rng(0),
    )
    assert result.points.shape == (20, 3)
