"""Tests for the safety layer (PROJECT-PLAN-V4 §11)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.safety import SafetyLimits, SafetyMonitor

DOF = 6


def make_limits() -> SafetyLimits:
    return SafetyLimits(
        dof=DOF,
        joint_range=np.array([[-3, 3]] * DOF, dtype=np.float64),
        max_joint_speed=np.full(DOF, 3.0),
        max_ctrl=np.full(DOF, 1.0),
        max_command_delta=np.full(DOF, 0.5),
    )


def make_monitor(locked=None) -> SafetyMonitor:
    return SafetyMonitor(make_limits(), locked_joints=locked)


def test_ok_command_accepted():
    mon = make_monitor()
    ok, v = mon.check_ctrl(np.zeros(DOF))
    assert ok
    assert v == []


def test_rejects_wrong_dim():
    mon = make_monitor()
    ok, v = mon.check_ctrl(np.zeros(DOF - 1))
    assert not ok
    assert v[0].code == "ctrl_dim"


def test_rejects_locked_joint_command():
    mon = make_monitor(locked=[2])
    command = np.zeros(DOF)
    command[2] = 0.5
    ok, v = mon.check_ctrl(command)
    assert not ok
    codes = {x.code for x in v}
    assert "locked_joint_command" in codes


def test_rejects_ctrl_above_cap():
    mon = make_monitor()
    ok, v = mon.check_ctrl(np.full(DOF, 1.5))
    assert not ok
    assert any(x.code == "ctrl_limit" for x in v)


def test_rejects_command_jump():
    mon = make_monitor()
    ok, _ = mon.check_ctrl(np.zeros(DOF), prev_ctrl=np.zeros(DOF))
    assert ok
    ok2, v2 = mon.check_ctrl(np.full(DOF, 0.9), prev_ctrl=np.zeros(DOF))
    assert not ok2
    assert any(x.code == "command_delta" for x in v2)


def test_state_joint_limit_and_speed():
    mon = make_monitor()
    qpos = np.zeros(DOF)
    qpos[0] = 4.0
    ok, v = mon.check_state(qpos, np.zeros(DOF))
    assert not ok
    assert v[0].code == "joint_limit"
    qvel = np.zeros(DOF)
    qvel[1] = 9.0
    ok2, v2 = mon.check_state(np.zeros(DOF), qvel)
    assert not ok2
    assert v2[0].code == "joint_speed"


def test_from_mapping_broadcast():
    limits = SafetyLimits.from_mapping(
        {
            "joint_range": [[-3, 3]] * DOF,
            "max_joint_speed": 2.0,  # scalar broadcast
            "max_ctrl": 1.0,
            "max_command_delta": 0.4,
        },
        dof=DOF,
    )
    assert limits.max_joint_speed.shape == (DOF,)
    assert np.allclose(limits.max_joint_speed, 2.0)
    assert np.allclose(limits.max_ctrl, 1.0)


def test_gate_returns_zero_on_breach():
    mon = make_monitor()
    safe, accepted, v = mon.gate(np.full(DOF, 2.0))
    assert not accepted
    assert np.all(safe == 0.0)
    assert any(x.code == "ctrl_limit" for x in v)


def test_gate_clips_delta_and_tracks():
    mon = make_monitor()
    # First accepted at zero, no last.
    safe, accepted, _ = mon.gate(np.zeros(DOF))
    assert accepted
    # Large jump gets clipped to max_command_delta, still accepted.
    safe2, accepted2, _ = mon.gate(np.full(DOF, 1.0))
    assert accepted2
    assert np.allclose(safe2, np.full(DOF, 0.5))


def test_flags_recordable():
    mon = make_monitor()
    qpos = np.zeros(DOF)
    qpos[0] = 5.0
    _ok, v = mon.check_state(qpos, np.zeros(DOF))
    flags = {}
    for x in v:
        flags.update(x.to_flag())
    assert flags["safety_joint_limit"] is True
