"""Tests for src/robotarm/envs/damage.py (DamageConfig)."""
from __future__ import annotations

import numpy as np
import pytest

from robotarm.envs.damage import D2, D3, D4, D5, DamageConfig, make_damage
from robotarm.envs.protocol import DamageConfig as DamageProtocol


def test_intact_has_no_locked_joints():
    d = DamageConfig.intact()
    assert d.n_locked == 0
    assert d.locked == []
    assert d.joint_mask.tolist() == [0, 0, 0, 0, 0]


def test_lock_single():
    d = DamageConfig.lock_single(2, 0.5)
    assert d.joint_mask.tolist() == [0, 0, 1, 0, 0]
    assert d.locked == [2]
    assert d.lock_angle_of(2) == pytest.approx(0.5)


def test_lock_single_out_of_range():
    with pytest.raises(ValueError):
        DamageConfig.lock_single(5, 0.0)
    with pytest.raises(ValueError):
        DamageConfig.lock_single(-1, 0.0)


def test_bad_mask_length():
    with pytest.raises(ValueError):
        DamageConfig(np.zeros(4), np.zeros(5))


def test_bad_mask_values():
    with pytest.raises(ValueError):
        DamageConfig(np.array([2, 0, 0, 0, 0]), np.zeros(5))


def test_bad_angle_length():
    with pytest.raises(ValueError):
        DamageConfig(np.zeros(5, dtype=int), np.zeros(4))


def test_equality_and_hash():
    a = DamageConfig.lock_single(2, 0.5)
    b = DamageConfig.lock_single(2, 0.5)
    c = DamageConfig.lock_single(2, 0.7)
    assert a == b
    assert a != c
    assert len({a, b, c}) == 2


def test_copy_is_independent():
    a = DamageConfig.lock_single(2, 0.5)
    b = a.copy()
    b.lock_angle[2] = 9.0
    assert a.lock_angle[2] == pytest.approx(0.5)


def test_satisfies_protocol_runtime_check():
    d = DamageConfig.lock_single(1, 0.3)
    assert isinstance(d, DamageProtocol)
    # Protocol attribute access works
    _ = d.joint_mask, d.lock_angle


def test_make_damage_styles():
    assert make_damage("intact").n_locked == 0
    assert make_damage("D2") == D2()
    assert make_damage("D3") == D3()
    assert make_damage("D5") == D5()
    with pytest.raises(KeyError):
        make_damage("D99")


def test_canonical_damages():
    # Damage labels follow physical joint numbers: D2 -> zero-based j2 index 1.
    assert D2().locked == [1]
    assert D3().locked == [2]
    assert D4().locked == [3]
    assert D5().locked == [4]
