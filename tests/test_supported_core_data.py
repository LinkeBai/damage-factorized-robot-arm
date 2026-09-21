"""Targeted identity, support, transition and non-overwrite checks for CPU data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from work.supported_core_compare import common, data


@pytest.fixture
def protocol():
    return {'data_seed': 91362026, 'steps': 50, 'segments': 5,
            'segment_steps': 10, 'action_limit': .8,
            'counts': {'pool': 40, 'validation': 20, 'test': 20},
            'workers': 1, 'shard_size': 250}


def test_global_mapping_balances_every_lock_and_profile():
    seen = {}
    for index in range(6000):
        lock, profile, local = data.identity_at(index, 6000)
        seen.setdefault((lock, profile), []).append(local)
    assert len(seen) == 20
    assert all(values == list(range(300)) for values in seen.values())
    with pytest.raises(ValueError):
        data.identity_at(0, 39)


def test_action_stream_is_identity_deterministic_and_split_separated(protocol):
    args = (2, 'mixed', 'pool', 101, 'random', protocol)
    actions = data.action_segments(*args)
    np.testing.assert_array_equal(actions, data.action_segments(*args))
    assert actions.shape == (5, 5)
    assert np.all(actions[:, 2] == 0.)
    assert np.max(abs(actions)) <= .8
    assert not np.array_equal(actions, data.action_segments(2, 'mixed', 'test', 101, 'random', protocol))
    assert not np.array_equal(actions, data.action_segments(2, 'mixed', 'pool', 102, 'random', protocol))


@pytest.mark.parametrize('kind,expected', [('zero', 0.), ('positive', .8), ('negative', -.8)])
def test_development_controls_exercise_all_free_motors(kind, expected, protocol):
    actions = data.action_segments(1, 'nominal', 'development', 0, kind, protocol)
    assert np.all(actions[:, 1] == 0.)
    assert np.all(actions[:, [0, 2, 3, 4]] == expected)


def test_actual_rollout_preserves_reset_identity_and_full_state(protocol, monkeypatch):
    calls = []
    original = common.make_reset
    def tracked(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(common, 'make_reset', tracked)
    row = data.run_trajectory(1, 'weak_motor', 'development', 0, 'random', protocol)
    assert calls == [(1, 'weak_motor', 'development', 0)]
    assert row['states'].shape == (51, 14)
    assert row['full_qpos'].shape == row['full_qvel'].shape == (51, 8)
    assert np.all(row['full_qvel'][0] == 0.)
    assert row['reset_record']['passed'] and row['diagnostics']['finite']
    assert row['diagnostics']['max_object_displacement_without_prior_contact_m'] == 0.
    assert not row['diagnostics']['dynamic_outcome_used_for_selection']
    assert row['diagnostics']['max_actuator_torque_error_Nm'] < 1e-12
    assert np.all((row['contacts'][:, 0] >= 1) & (row['contacts'][:, 0] <= 50))
    mapping = {name: i for i, name in enumerate(row['reset_record']['joint_order'])}
    np.testing.assert_allclose(row['states'][:, :5], row['full_qpos'][:, [mapping[f'j{i}'] for i in range(1, 6)]], atol=1e-7)


def test_invalid_initial_state_cannot_enter_dynamic_rollout(protocol, monkeypatch):
    model, initial, record = common.make_reset(0, 'nominal', 'development', 0)
    initial.qvel[0] = .1
    monkeypatch.setattr(common, 'make_reset', lambda *args: (model, initial, record))
    with pytest.raises(RuntimeError, match='Initial geometry/support hard gate failed'):
        data.run_trajectory(0, 'nominal', 'development', 0, 'random', protocol)


def test_shard_preserves_records_and_refuses_overwrite(tmp_path, protocol):
    row = data.run_trajectory(0, 'nominal', 'development', 0, 'zero', protocol)
    target = tmp_path / 'shard'
    manifest = data.store_rows(target, [row], {'split': 'development'})
    assert manifest['npz_sha256'] == data.file_sha(target / 'data.npz')
    assert manifest['records_sha256'] == data.file_sha(target / 'records.json')
    with np.load(target / 'data.npz', allow_pickle=False) as shard:
        assert shard['states'].shape == (1, 51, 14)
        assert shard['contacts'].shape[1] == 8
    assert json.loads((target / 'records.json').read_text())[0]['reset_record']['passed']
    with pytest.raises(FileExistsError):
        data.store_rows(target, [row], {'split': 'development'})


def test_resume_checks_exact_file_hash_and_current_source(tmp_path, protocol, monkeypatch):
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
    monkeypatch.setattr(common, 'OUT', tmp_path)
    monkeypatch.setattr(common, 'PROTOCOL', protocol_path)
    row = data.run_trajectory(0, 'nominal', 'development', 0, 'zero', protocol)
    target = tmp_path / 'data' / 'pool' / 'shard-000000-000001'
    data.store_rows(target, [row], {'split': 'pool', 'start': 0, 'end': 1,
        'protocol_sha256': data.protocol_hash(protocol), 'source_sha256': data.source_hashes()})
    assert data.validate_shard(target, protocol)['count'] == 1
    assert data.discover_shards('pool', protocol)[0]['npz_path'] == str((target / 'data.npz').resolve())
    with (target / 'data.npz').open('ab') as stream:
        stream.write(b'deliberate corruption')
    with pytest.raises(RuntimeError, match='Corrupt shard hash'):
        data.validate_shard(target, protocol)


def test_failed_physics_gate_blocks_collection_before_submission(tmp_path, protocol, monkeypatch):
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
    monkeypatch.setattr(common, 'OUT', tmp_path)
    monkeypatch.setattr(common, 'PROTOCOL', protocol_path)
    (tmp_path / 'physics-gate.json').write_text(json.dumps({'passed': False,
        'protocol_sha256': data.file_sha(protocol_path)}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='physics gate has not passed'):
        data.collect(protocol, 'pool')
    assert not (tmp_path / 'data').exists()


def test_internal_substeps_keep_fixed_observation_horizon_and_capture_extrema(protocol, monkeypatch):
    import mujoco
    original = common.make_reset
    def fine_model(*args):
        model, initial, record = original(*args)
        model.opt.timestep = .001
        mujoco.mj_forward(model, initial)
        return model, initial, record
    monkeypatch.setattr(common, 'make_reset', fine_model)
    row = data.run_trajectory(1, 'weak_motor', 'development', 0, 'random', protocol)
    internal = row['substep_diagnostic_metrics']
    assert internal.shape == (50, 5, len(data.METRIC_NAMES))
    np.testing.assert_array_equal(row['diagnostic_metrics'][1:], internal[:, -1])
    assert row['substep_time_s'].shape == (50, 5)
    assert row['substep_time_s'][-1, -1] == pytest.approx(.25)
    assert np.all((row['contacts'][:, 1] >= 1) & (row['contacts'][:, 1] <= 5))
    column = data.METRIC_NAMES.index('min_arm_table_m')
    expected_min = min(row['diagnostic_metrics'][0, column], internal[:, :, column].min())
    assert row['diagnostics']['min_arm_table_m'] == expected_min


def test_runtime_failure_keeps_failed_trajectory_snapshot(protocol, monkeypatch):
    def broken_step(model, initial, on_substep=None):
        raise RuntimeError('deliberate execution failure')
    monkeypatch.setattr(common, 'step', broken_step)
    with pytest.raises(data.TrajectoryExecutionError) as caught:
        data.run_trajectory(0, 'nominal', 'development', 0, 'zero', protocol)
    snapshot = caught.value.snapshot
    assert snapshot['failed_observation_step'] == 1
    assert len(snapshot['qpos']) == len(snapshot['qvel']) == 8
    assert np.asarray(snapshot['states_before_failure']).shape == (1, 14)
    assert snapshot['reset_record']['passed']
