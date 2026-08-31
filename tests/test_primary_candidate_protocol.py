from pathlib import Path

import numpy as np
import pytest
import mujoco

from scripts.verify_primary_candidate_protocol import audit_candidate_file
from scripts.generate_primary_sequence_candidates import contact_geom_pairs


def _write(path: Path, *, candidates: int, locks=(1, 3), duplicate=False) -> None:
    groups, joint, actions = [], [], []
    rng = np.random.default_rng(5)
    for group, locked in enumerate(locks):
        group_actions = rng.normal(size=(candidates, 5, 5))
        group_actions[:, :, locked] = 0.0
        if duplicate:
            group_actions[-1] = group_actions[0]
        groups.extend([group] * candidates)
        joint.extend([locked] * candidates)
        actions.extend(group_actions)
    np.savez(path, group=np.asarray(groups), locked_joint=np.asarray(joint),
             action_sequence=np.asarray(actions))


def test_formal_evaluation_requires_128_unique_candidates(tmp_path):
    path = tmp_path / "candidates.npz"
    _write(path, candidates=128)
    result = audit_candidate_file(
        path, allowed_locks={1, 3}, expected_candidates=128,
        horizon_steps=50, steps_per_segment=10)
    assert result["candidates_per_group"] == 128
    assert result["duplicate_candidate_groups"] == 0


def test_repeating_32_candidates_cannot_pass_as_128(tmp_path):
    path = tmp_path / "candidates.npz"
    _write(path, candidates=128, duplicate=True)
    with pytest.raises(ValueError, match="duplicated"):
        audit_candidate_file(
            path, allowed_locks={1, 3}, expected_candidates=128,
            horizon_steps=50, steps_per_segment=10)


def test_training_can_use_32_candidates_and_filters_d3(tmp_path):
    path = tmp_path / "candidates.npz"
    _write(path, candidates=32, locks=(1, 3, 2))
    result = audit_candidate_file(
        path, allowed_locks={1, 3}, expected_candidates=None,
        horizon_steps=50, steps_per_segment=10)
    assert result["selected_locks"] == [1, 3]
    assert result["rows_excluded_by_lock_filter"] == 32


def test_confirmation_accepts_only_d3_rows(tmp_path):
    path = tmp_path / "candidates.npz"
    _write(path, candidates=128, locks=(1, 3, 2))
    result = audit_candidate_file(
        path, allowed_locks={2}, expected_candidates=128,
        horizon_steps=50, steps_per_segment=10)
    assert result["selected_locks"] == [2]
    assert result["rows_excluded_by_lock_filter"] == 256


def test_candidate_generator_monitors_tool_and_pusher_contacts():
    model = mujoco.MjModel.from_xml_path("sim/assets/arm_push.xml")
    pairs = contact_geom_pairs(model)
    block = int(model.geom("block_geom").id)
    assert frozenset((int(model.geom("tool_geom").id), block)) in pairs
    assert frozenset((int(model.geom("pusher_geom").id), block)) in pairs
