"""Tests for task definitions (PROJECT-PLAN-V4 §5.1–5.3)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.tasks import PickTask, PushTask, ReachTask, build_task

TARGET = np.array([0.3, 0.1, 0.2])


def make_state(ee=None, obj=None, grasped=False):
    st = {}
    if ee is not None:
        st["ee_pos"] = np.asarray(ee, dtype=np.float64)
    if obj is not None:
        st["obj_pos"] = np.asarray(obj, dtype=np.float64)
    st["grasped"] = grasped
    return st


class TestReach:
    def test_success_within_tolerance(self):
        t = ReachTask(tolerance=0.05)
        assert t.success(make_state(ee=TARGET), TARGET)
        assert not t.success(make_state(ee=TARGET + np.array([1.0, 0, 0])), TARGET)

    def test_reward_reflects_distance(self):
        t = ReachTask()
        far = make_state(ee=TARGET + np.array([1.0, 0, 0]))
        near = make_state(ee=TARGET)
        r_far = t.reward(far, TARGET)
        r_near = t.reward(near, TARGET)
        assert r_near > r_far
        assert r_near == pytest.approx(5.0, abs=1e-9)  # bonus when within tol


class TestPush:
    def test_success(self):
        t = PushTask(tolerance=0.05)
        assert t.success(make_state(obj=TARGET), TARGET)
        assert not t.success(make_state(obj=TARGET + [0.5, 0, 0]), TARGET)

    def test_reward_bonus(self):
        t = PushTask()
        assert t.reward(make_state(obj=TARGET), TARGET) == pytest.approx(5.0)


class TestPick:
    def test_needs_grasp(self):
        t = PickTask()
        assert t.success(make_state(obj=TARGET, grasped=True), TARGET)
        assert not t.success(make_state(obj=TARGET, grasped=False), TARGET)

    def test_reward(self):
        t = PickTask()
        assert t.reward(make_state(obj=TARGET, grasped=True), TARGET) == pytest.approx(10.0)


def test_build_task():
    assert isinstance(build_task("reach"), ReachTask)
    assert isinstance(build_task("push"), PushTask)
    assert isinstance(build_task("pick"), PickTask)
    assert build_task("reach", tolerance=0.09).tolerance == pytest.approx(0.09)
    with pytest.raises(KeyError):
        build_task("nope")