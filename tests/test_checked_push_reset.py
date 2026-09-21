import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import pytest

from robotarm.envs.checked_push_reset import JOINTS, PROFILES, inspect_reset, make_model, planning_reset, sample_reset
from robotarm.envs.constraint_lock import activate_joint_lock


@pytest.mark.parametrize('lock', range(5))
def test_reset_is_legal_deterministic_and_order_independent(lock):
    m = make_model(lock, PROFILES[lock % 4])
    d, record = sample_reset(m, lock, PROFILES[lock % 4], 'development', 3)
    expected = d.qpos.copy()
    sample_reset(m, lock, PROFILES[lock % 4], 'development', 19)
    again, r = sample_reset(m, lock, PROFILES[lock % 4], 'development', 3)
    assert np.array_equal(expected, again.qpos)
    assert r == record
    assert inspect_reset(m, again, lock)['passed']


def test_detects_old_penetrating_reset_and_out_of_range_lock():
    m = make_model(4, 'nominal'); d = mujoco.MjData(m)
    q = [.34396525, .79659639, .79649504, 1.42124113, 1.59]
    for n, v in zip(JOINTS, q): d.qpos[int(m.joint(n).qposadr[0])] = v
    activate_joint_lock(m, d, 'j5', q[4])
    audit = inspect_reset(m, d, 4)
    assert not audit['passed']
    assert {'joint_margin', 'arm_table_clearance', 'arm_block_clearance'} <= set(audit['reasons'])


def test_wrong_lock_and_velocity_are_rejected():
    m = make_model(2, 'nominal'); d, _ = sample_reset(m, 2, 'nominal', 'development', 0)
    d.qvel[int(m.joint('j3').dofadr[0])] = .1
    d.eq_active[:] = False
    reasons = inspect_reset(m, d, 2)['reasons']
    assert 'nonzero_locked_velocity' in reasons and 'wrong_active_lock' in reasons


def test_planning_uses_checked_reset_and_separate_bands():
    for band in (0, 1):
        m, d, goal, r = planning_reset(2, 'mixed', band, 4)
        assert inspect_reset(m, d, 2)['passed']
        lo, hi = ((.04, .065), (.065, .09))[band]
        assert lo <= np.linalg.norm(goal-d.body('block').xpos[:2]) <= hi
    assert 'planning' in r['reset_id']
