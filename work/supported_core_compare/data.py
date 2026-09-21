"""CPU-only supported-push data and physics audit; no outcome filtering or training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mujoco
import numpy as np
from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported
from robotarm.envs.constraint_lock import activate_joint_lock
from work.supported_core_compare import common

PROFILES = checked.PROFILES
SPLIT_CODES = {'development': 1, 'pool': 101, 'validation': 202, 'test': 303}
METRIC_NAMES = (
    'min_joint_margin_rad', 'max_lock_drift_rad', 'max_lock_speed_rad_s',
    'min_arm_table_m', 'min_arm_block_m', 'nonassembly_self_min_m',
    'block_table_m', 'block_z_m', 'block_vz_m_s', 'support_normal_force_N',
    'support_normal_error_N', 'object_displacement_m', 'max_abs_arm_speed_rad_s',
    'max_actuator_torque_error_Nm', 'finite',
) + tuple('table_' + name + '_m' for name in checked.ARM_GEOMS)


class TrajectoryExecutionError(RuntimeError):
    def __init__(self, message, snapshot):
        super().__init__(message)
        self.snapshot = snapshot


def settings(protocol):
    return {'data_seed': int(protocol.get('data_seed', 91362026)),
        'steps': int(protocol.get('steps', 50)), 'segments': int(protocol.get('segments', 5)),
        'segment_steps': int(protocol.get('segment_steps', 10)),
        'action_limit': float(protocol.get('action_limit', .8)),
        'counts': protocol.get('counts', {'pool': 50000, 'validation': 2000, 'test': 6000}),
        'workers': int(protocol.get('workers', 4)), 'shard_size': int(protocol.get('shard_size', 250))}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protocol_hash(protocol):
    """All new protocol_sha256 fields bind the exact frozen file bytes."""
    if common.read(common.PROTOCOL) != protocol:
        raise ValueError('In-memory protocol differs from frozen protocol file')
    return file_sha(common.PROTOCOL)


def source_hashes():
    paths = [Path(__file__), Path(common.__file__), Path(supported.__file__),
             Path(checked.__file__), checked.XML,
             ROOT / 'src/robotarm/envs/constraint_lock.py']
    if common.load_protocol().get('numerics'):
        paths.append(ROOT / 'src/robotarm/envs/resolved_push_reset.py')
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def identity_at(global_index, total):
    if total <= 0 or total % 20 or not 0 <= global_index < total:
        raise ValueError('Split must contain equally sized 20 lock/profile cells')
    per_cell = total // 20
    cell, index = divmod(global_index, per_cell)
    lock, pi = divmod(cell, len(PROFILES))
    return lock, PROFILES[pi], index


def action_segments(lock, profile, split, index, kind, protocol):
    cfg = settings(protocol)
    if split not in SPLIT_CODES or profile not in PROFILES or lock not in range(5) or index < 0:
        raise ValueError('Invalid trajectory identity')
    actions = np.zeros((cfg['segments'], 5), dtype=np.float64)
    free = [i for i in range(5) if i != lock]
    if kind == 'random':
        rng = np.random.default_rng(np.random.SeedSequence([cfg['data_seed'],
            SPLIT_CODES[split], lock, PROFILES.index(profile), index, 917]))
        actions[:] = rng.uniform(-cfg['action_limit'], cfg['action_limit'], actions.shape)
    elif kind in ('positive', 'negative'):
        actions[:, free] = cfg['action_limit'] * (1 if kind == 'positive' else -1)
    elif kind != 'zero':
        raise ValueError('Unknown control kind')
    actions[:, lock] = 0.
    return actions


def contact_records(model, data):
    """Call immediately after mj_step, before mj_forward, for actual transition forces."""
    rows = []
    for ci, contact in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, force)
        rows.append({'geom1': int(contact.geom1), 'geom2': int(contact.geom2),
            'geom1_name': model.geom(int(contact.geom1)).name,
            'geom2_name': model.geom(int(contact.geom2)).name,
            'distance_m': float(contact.dist), 'normal_force_N': float(force[0]),
            'tangent_force_norm_N': float(np.linalg.norm(force[1:3]))})
    return rows


def frame_diagnostics(model, data, lock, initial_state):
    """Post-forward geometry explicitly includes visual geoms with collisions off."""
    geo = checked.geometry(model, data)
    support = supported.support_measurement(model, data)
    state = common.state14(model, data)
    av = [int(model.joint(n).dofadr[0]) for n in checked.JOINTS]
    desired = np.clip(data.ctrl, model.actuator_forcerange[:, 0],
                      model.actuator_forcerange[:, 1]) * model.actuator_gear[:, 0]
    result = {'min_joint_margin_rad': min(geo['joint_margins_rad'].values()),
        'max_lock_drift_rad': float(abs(state[lock] - initial_state[lock])),
        'max_lock_speed_rad_s': float(abs(state[5 + lock])),
        'min_arm_table_m': min(v for n, v in geo['arm_table_m'].items() if n != 'base_geom'),
        'min_arm_block_m': min(geo['arm_block_m'].values()),
        'nonassembly_self_min_m': geo['nonassembly_self_min_m'],
        'block_table_m': geo['block_table_m'], 'block_z_m': support['z_displacement_m'],
        'block_vz_m_s': support['z_velocity_m_s'],
        'support_normal_force_N': support['normal_force_N'],
        'support_normal_error_N': support['normal_error_N'],
        'object_displacement_m': float(np.linalg.norm(state[10:12] - initial_state[10:12])),
        'max_abs_arm_speed_rad_s': float(np.max(abs(data.qvel[av]))),
        'max_actuator_torque_error_Nm': float(np.max(abs(data.qfrc_actuator[av] - desired))),
        'finite': bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                       and np.isfinite(data.qacc).all())}
    result.update({'table_' + n + '_m': v for n, v in geo['arm_table_m'].items()})
    return result


def run_trajectory(lock, profile, split, index, kind, protocol):
    cfg = settings(protocol)
    if cfg['steps'] != cfg['segments'] * cfg['segment_steps']:
        raise ValueError('Trajectory length must equal segment count times segment length')
    model, data, reset = common.make_reset(lock, profile, split, index)
    audit = supported.inspect_reset(model, data, lock)
    if not audit['passed']:
        raise RuntimeError(f'Initial geometry/support hard gate failed: {audit}')
    actions = action_segments(lock, profile, split, index, kind, protocol)
    n = cfg['steps'] + 1
    states = np.empty((n, 14), dtype=np.float64)
    qpos = np.empty((n, model.nq), dtype=np.float64)
    qvel = np.empty((n, model.nv), dtype=np.float64)
    metrics = np.empty((n, len(METRIC_NAMES)), dtype=np.float64)
    substeps = int(round(.005 / model.opt.timestep))
    if substeps < 1 or abs(substeps * model.opt.timestep - .005) > 1e-12:
        raise ValueError('Internal timestep must divide the fixed 5ms observation interval')
    internal_metrics = np.empty((cfg['steps'], substeps, len(METRIC_NAMES)), dtype=np.float64)
    internal_times = np.empty((cfg['steps'], substeps), dtype=np.float64)
    start = np.asarray(common.state14(model, data)).copy()
    actual_contacts = []
    ever_object_contact = False
    first_contact_step = None
    first_contact_substep = None
    max_before_contact = 0.
    actuator_impulse = 0.
    contacts_per_pair = {}
    eq = model.equality('fault_lock_' + checked.JOINTS[lock]).id
    expected_lock = reset['lock_angle_rad']
    def on_substep(current_model, current_data, internal_index):
        nonlocal ever_object_contact, first_contact_step, first_contact_substep
        nonlocal actuator_impulse, max_before_contact
        if not 0 <= internal_index < substeps:
            raise RuntimeError('Unexpected internal callback index')
        for row in contact_records(current_model, current_data):
            actual_contacts.append([step, internal_index + 1, row['geom1'], row['geom2'], row['distance_m'],
                                    row['normal_force_N'], row['tangent_force_norm_N']])
            pair = '/'.join(sorted((row['geom1_name'], row['geom2_name'])))
            entry = contacts_per_pair.setdefault(pair, {'contact_point_samples': 0, 'max_normal_N': 0.,
                'normal_impulse_Ns': 0., 'min_distance_m': 1., 'internal_indices': set(), 'observation_indices': set()})
            entry['contact_point_samples'] += 1
            entry['internal_indices'].add((step, internal_index))
            entry['observation_indices'].add(step)
            entry['max_normal_N'] = max(entry['max_normal_N'], row['normal_force_N'])
            entry['normal_impulse_Ns'] += row['normal_force_N'] * current_model.opt.timestep
            entry['min_distance_m'] = min(entry['min_distance_m'], row['distance_m'])
            if ('block_geom' in (row['geom1_name'], row['geom2_name'])
                    and {row['geom1_name'], row['geom2_name']} & {'tool_geom', 'pusher_geom'}
                    and row['normal_force_N'] > 1e-9):
                ever_object_contact = True
                if first_contact_step is None:
                    first_contact_step, first_contact_substep = step, internal_index + 1
        actuator_impulse += float(np.linalg.norm(current_data.qfrc_actuator)) * current_model.opt.timestep
        mujoco.mj_forward(current_model, current_data)
        diag = frame_diagnostics(current_model, current_data, lock, start)
        internal_metrics[step - 1, internal_index] = [diag[name] for name in METRIC_NAMES]
        internal_times[step - 1, internal_index] = current_data.time
        if not ever_object_contact:
            max_before_contact = max(max_before_contact, diag['object_displacement_m'])
    for step in range(n):
        if step:
            data.ctrl[:] = actions[(step - 1) // cfg['segment_steps']]
            # This model is owned by this trajectory; still defend the lock target.
            assert model.eq_data[eq, 0] == expected_lock
            assert np.count_nonzero(data.eq_active) == 1 and data.eq_active[eq]
            try:
                common.step(model, data, on_substep=on_substep)
            except Exception as error:
                raise TrajectoryExecutionError('Integration/capture failed; partial state preserved', {
                    'identity': [lock, profile, split, index, kind], 'failed_observation_step': step,
                    'time_s': float(data.time), 'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
                    'ctrl': data.ctrl.tolist(), 'states_before_failure': states[:step].tolist(),
                    'full_qpos_before_failure': qpos[:step].tolist(), 'full_qvel_before_failure': qvel[:step].tolist(),
                    'actual_contacts_before_failure': actual_contacts,
                    'reset_record': reset}) from error
        else:
            mujoco.mj_forward(model, data)
        states[step] = common.state14(model, data)
        qpos[step], qvel[step] = data.qpos, data.qvel
        if step:
            metrics[step] = internal_metrics[step - 1, -1]
        else:
            initial_diagnostics = frame_diagnostics(model, data, lock, start)
            metrics[step] = [initial_diagnostics[name] for name in METRIC_NAMES]
    if abs(data.time - cfg['steps'] * .005) > 1e-10:
        raise RuntimeError('Observation interval changed: rollout must preserve the frozen physical horizon')
    all_metrics = np.concatenate((metrics[:1], internal_metrics.reshape(-1, len(METRIC_NAMES))))
    def column(name):
        return all_metrics[:, METRIC_NAMES.index(name)]
    for entry in contacts_per_pair.values():
        entry['substeps_with_contact'] = len(entry.pop('internal_indices'))
        entry['observation_intervals_with_contact'] = len(entry.pop('observation_indices'))
    diagnostics = {'finite': bool(column('finite').all()),
        'first_actual_object_contact_step': first_contact_step,
        'first_actual_object_contact_substep': first_contact_substep,
        'contact_step_index_base': 1, 'contact_substep_index_base': 1,
        'internal_timestep_s': model.opt.timestep, 'internal_substeps_per_observation': substeps,
        'observation_steps': cfg['steps'], 'final_time_s': float(data.time),
        'max_object_displacement_without_prior_contact_m': max_before_contact,
        'object_displacement_m': float(column('object_displacement_m')[-1]),
        'max_actuator_torque_error_Nm': float(column('max_actuator_torque_error_Nm').max()),
        'actuator_generalized_force_norm_integral_Ns': actuator_impulse,
        'min_joint_margin_rad': float(column('min_joint_margin_rad').min()),
        'max_lock_drift_rad': float(column('max_lock_drift_rad').max()),
        'min_arm_table_m': float(column('min_arm_table_m').min()),
        'min_arm_block_m': float(column('min_arm_block_m').min()),
        'min_nonassembly_self_m': float(column('nonassembly_self_min_m').min()),
        'min_block_table_m': float(column('block_table_m').min()),
        'z_range_m': [float(column('block_z_m').min()), float(column('block_z_m').max())],
        'max_abs_vz_m_s': float(np.max(abs(column('block_vz_m_s')))),
        'support_normal_range_N': [float(column('support_normal_force_N').min()), float(column('support_normal_force_N').max())],
        'arm_table_minima_m': {name: float(column('table_' + name + '_m').min()) for name in checked.ARM_GEOMS},
        'contacts_per_pair': contacts_per_pair,
        'dynamic_outcome_used_for_selection': False}
    diagnostics['mujoco_warning_counts'] = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    bad_warnings = [int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL), int(mujoco.mjtWarning.mjWARN_BADQACC)]
    diagnostics['finite'] = diagnostics['finite'] and not any(data.warning[i].number for i in bad_warnings)
    initial_sha = hashlib.sha256(np.concatenate((qpos[0], qvel[0])).astype('<f8').tobytes()).hexdigest()
    return {'states': states, 'full_qpos': qpos, 'full_qvel': qvel,
        'segment_actions': actions, 'diagnostic_metrics': metrics,
        'substep_diagnostic_metrics': internal_metrics, 'substep_time_s': internal_times,
        'contacts': np.asarray(actual_contacts, dtype=np.float64).reshape(-1, 7),
        'reset_record': reset, 'initial_state_sha256': initial_sha,
        'diagnostics': diagnostics, 'lock': lock, 'profile': profile,
        'split': split, 'index': index, 'control_kind': kind}


def trajectory_record(row):
    return {k: row[k] for k in ('reset_record', 'initial_state_sha256', 'diagnostics',
        'lock', 'profile', 'split', 'index', 'control_kind')}


def store_rows(directory, rows, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    contacts = []
    for index, row in enumerate(rows):
        if len(row['contacts']):
            contacts.append(np.column_stack((np.full(len(row['contacts']), index), row['contacts'])))
    arrays = {name: np.stack([row[name] for row in rows]) for name in
        ('states', 'full_qpos', 'full_qvel', 'segment_actions', 'diagnostic_metrics', 'substep_diagnostic_metrics', 'substep_time_s')}
    arrays.update({'contacts': np.concatenate(contacts) if contacts else np.empty((0, 8)),
        'reset_id': np.array([row['reset_record']['reset_id'] for row in rows]),
        'initial_state_sha256': np.array([row['initial_state_sha256'] for row in rows]),
        'locked_joint': np.array([row['lock'] for row in rows], dtype=np.int8),
        'profile': np.array([row['profile'] for row in rows]),
        'control_kind': np.array([row['control_kind'] for row in rows])})
    with (directory / 'data.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_new(directory / 'records.json', [trajectory_record(row) for row in rows])
    manifest = {**metadata, 'count': len(rows), 'npz_sha256': file_sha(directory / 'data.npz'),
        'records_sha256': file_sha(directory / 'records.json'),
        'metric_names': METRIC_NAMES,
        'contacts_columns': ['trajectory', 'observation_step', 'internal_substep', 'geom1', 'geom2', 'distance_m', 'normal_force_N', 'tangent_force_norm_N'],
        'contact_step_and_substep_index_base': 1,
        'contact_timing': 'Actual solver contact from mj_step before post-step mj_forward',
        'schema': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in arrays.items()},
        'hard_gate_passed': all(row['reset_record']['passed'] and row['diagnostics']['finite'] for row in rows),
        'selection': 'Every requested identity retained; no filtering by dynamic outcome'}
    write_new(directory / 'manifest.json', manifest)
    return manifest


def summarize(rows):
    ds = [row['diagnostics'] for row in rows]
    return {'trajectories': len(rows), 'hard_gate_passed': all(row['reset_record']['passed'] and row['diagnostics']['finite'] for row in rows),
        'min_arm_table_m': min(d['min_arm_table_m'] for d in ds),
        'min_arm_block_m': min(d['min_arm_block_m'] for d in ds),
        'min_joint_margin_rad': min(d['min_joint_margin_rad'] for d in ds),
        'min_block_table_m': min(d['min_block_table_m'] for d in ds),
        'z_range_m': [min(d['z_range_m'][0] for d in ds), max(d['z_range_m'][1] for d in ds)],
        'max_abs_vz_m_s': max(d['max_abs_vz_m_s'] for d in ds),
        'max_object_displacement_without_prior_contact_m': max(d['max_object_displacement_without_prior_contact_m'] for d in ds),
        'arm_table_minima_m': {name: min(d['arm_table_minima_m'][name] for d in ds) for name in checked.ARM_GEOMS},
        'contact_trajectories': sum(d['first_actual_object_contact_step'] is not None for d in ds),
        'object_displacement_range_m': [min(d['object_displacement_m'] for d in ds), max(d['object_displacement_m'] for d in ds)],
        'max_actuator_torque_error_Nm': max(d['max_actuator_torque_error_Nm'] for d in ds),
        'dynamic_soft_contact_threshold_applied': False}


def development(protocol):
    output = Path(common.OUT) / protocol.get('development_directory', 'data-development')
    report_path = Path(common.OUT) / protocol.get('development_report', 'development-report.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(f'Preserve previous development evidence: {output}')
    rows = []
    error = None
    before = source_hashes()
    try:
        for lock in range(5):
            for profile in PROFILES:
                for index in range(4):
                    for kind in ('zero', 'random', 'positive', 'negative'):
                        rows.append(run_trajectory(lock, profile, 'development', index, kind, protocol))
                print(f'Development completed D{lock + 1} {profile}', flush=True)
    except Exception as failure:
        error = {'identity': [lock, profile, index, kind], 'traceback': traceback.format_exc()}
        if hasattr(failure, 'snapshot'):
            error['failed_trajectory_snapshot'] = failure.snapshot
    if not rows:
        write_new(report_path, {'hard_gate_passed': False, 'error': error,
            'protocol_sha256': protocol_hash(protocol), 'source_sha256': before})
        raise RuntimeError('Development failed before first trajectory; failure preserved')
    summary = summarize(rows)
    summary.update({'hard_gate_passed': summary['hard_gate_passed'] and error is None and len(rows) == 320,
        'error': error, 'scope': 'Formal-action development physics gate; root independently reviews dynamics',
        'source_sha256': before, 'source_unchanged_during_run': before == source_hashes(),
        'protocol_sha256': protocol_hash(protocol), 'raw_directory': str(output),
        'physics_gate_written_by_this_script': False})
    metadata = {'scope': 'Formal-action development physics gate, not model performance',
        'protocol_sha256': protocol_hash(protocol), 'source_sha256': before,
        'settings': settings(protocol), 'split': 'development', 'summary': summary}
    store_rows(output, rows, metadata)
    write_new(output / 'summary.json', summary)
    write_new(report_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['hard_gate_passed']:
        raise RuntimeError('Development hard gate failed; all completed trajectories retained')


def validate_shard(directory, protocol=None, *, verify=True):
    """Reject corruption or identity changes; never regenerate over a prior shard."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = common.read(manifest_path)
    if protocol is not None:
        if manifest['protocol_sha256'] != protocol_hash(protocol):
            raise RuntimeError(f'Protocol hash mismatch: {directory}')
        if manifest['source_sha256'] != source_hashes():
            raise RuntimeError(f'Implementation/source hash mismatch: {directory}')
    if not manifest.get('hard_gate_passed'):
        raise RuntimeError(f'Prior shard failed its hard gate: {directory}')
    if verify:
        for filename, key in [('data.npz', 'npz_sha256'), ('records.json', 'records_sha256')]:
            if file_sha(directory / filename) != manifest[key]:
                raise RuntimeError(f'Corrupt shard hash: {directory / filename}')
        with np.load(directory / 'data.npz', allow_pickle=False) as arrays:
            for name, description in manifest['schema'].items():
                if list(arrays[name].shape) != description['shape'] or str(arrays[name].dtype) != description['dtype']:
                    raise RuntimeError(f'Unexpected array schema: {directory}/{name}')
            for name in ('states', 'full_qpos', 'full_qvel', 'segment_actions', 'diagnostic_metrics', 'substep_diagnostic_metrics', 'substep_time_s', 'contacts'):
                if not np.isfinite(arrays[name]).all():
                    raise RuntimeError(f'Nonfinite array: {directory}/{name}')
            if len(arrays['states']) != manifest['count']:
                raise RuntimeError(f'Shard count mismatch: {directory}')
    return {**manifest, 'manifest_path': str(manifest_path.resolve()),
        'manifest_sha256': file_sha(manifest_path),
        'npz_path': str((directory / 'data.npz').resolve()),
        'records_path': str((directory / 'records.json').resolve())}


def discover_shards(split, protocol=None, verify=True):
    if split not in ('pool', 'validation', 'test'):
        raise ValueError('Expected pool, validation or test split')
    base = Path(common.OUT) / 'data' / split
    return [validate_shard(path.parent, protocol, verify=verify)
        for path in sorted(base.glob('shard-*/manifest.json'))]


def collect_shard(task):
    split, start, end, protocol, expected_sources = task
    total = int(settings(protocol)['counts'][split])
    parent = Path(common.OUT) / 'data' / split
    parent.mkdir(parents=True, exist_ok=True)
    stem = f'shard-{start:06d}-{end:06d}'
    final = parent / stem
    if final.exists():
        return {'status': 'resumed', **validate_shard(final, protocol)}
    if source_hashes() != expected_sources:
        raise RuntimeError('Source changed before shard execution')
    incomplete = parent / ('.incomplete-' + stem + '-' + uuid.uuid4().hex)
    rows = []
    try:
        for global_index in range(start, end):
            lock, profile, index = identity_at(global_index, total)
            row = run_trajectory(lock, profile, split, index, 'random', protocol)
            rows.append(row)
        metadata = {'split': split, 'start': start, 'end': end,
            'protocol_sha256': protocol_hash(protocol), 'source_sha256': expected_sources,
            'settings': settings(protocol), 'summary': summarize(rows)}
        manifest = store_rows(incomplete, rows, metadata)
        if not manifest['hard_gate_passed']:
            raise RuntimeError('Nonfinite/invalid trajectory retained; this shard cannot commit')
        if source_hashes() != expected_sources:
            raise RuntimeError('Source changed during shard execution')
        if final.exists():
            raise FileExistsError(f'Concurrent shard already exists; retain incomplete evidence: {final}')
        incomplete.rename(final)
        return {'status': 'created', **validate_shard(final, protocol)}
    except Exception as failure:
        if not incomplete.exists() and rows:
            store_rows(incomplete, rows, {'split': split, 'start': start, 'end': end,
                'protocol_sha256': protocol_hash(protocol), 'source_sha256': expected_sources,
                'incomplete': True})
        incomplete.mkdir(parents=True, exist_ok=True)
        write_new(incomplete / 'failure.json', {'split': split, 'start': start, 'end': end,
            'completed_trajectories': len(rows), 'error': traceback.format_exc(),
            'failed_trajectory_snapshot': getattr(failure, 'snapshot', None),
            'protocol_sha256': protocol_hash(protocol), 'source_sha256': expected_sources})
        raise


def require_physics_gate(protocol):
    gate = common.read(Path(common.OUT) / 'physics-gate.json')
    if not gate.get('passed'):
        raise RuntimeError('Independent physics gate has not passed')
    if gate.get('protocol_sha256') != protocol_hash(protocol):
        raise RuntimeError('Physics gate does not bind the current frozen protocol')
    return gate


def collect(protocol, split):
    if split not in ('pool', 'validation', 'test'):
        raise ValueError('--split is required for collection')
    require_physics_gate(protocol)
    cfg = settings(protocol)
    if cfg['workers'] not in range(1, 5) or cfg['shard_size'] not in (250, 500):
        raise ValueError('Collection is limited to 1-4 CPU workers and 250/500 trajectories per shard')
    total = int(cfg['counts'][split])
    identity_at(0, total)
    sources = source_hashes()
    tasks = [(split, start, min(total, start + cfg['shard_size']), protocol, sources)
        for start in range(0, total, cfg['shard_size'])]
    completed, failures = [], []
    # Existing shards are validated before any new simulation is submitted.
    pending = []
    for task in tasks:
        _, start, end, _, _ = task
        path = Path(common.OUT) / 'data' / split / f'shard-{start:06d}-{end:06d}'
        if path.exists():
            completed.append({'status': 'resumed', **validate_shard(path, protocol)})
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=cfg['workers']) as executor:
        futures = {executor.submit(collect_shard, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
                completed.append(row)
                print(json.dumps({'split': split, 'start': row['start'], 'end': row['end'], 'status': row['status']}), flush=True)
            except Exception:
                failures.append({'split': split, 'start': task[1], 'end': task[2], 'error': traceback.format_exc()})
                for other in futures:
                    other.cancel()
                # Running workers finish their bounded shards; every output is retained.
                break
    report = {'split': split, 'expected_count': total, 'completed_shards': len(completed),
        'failures': failures, 'protocol_sha256': protocol_hash(protocol), 'source_sha256': sources,
        'engine': 'MuJoCo CPU', 'workers': cfg['workers'], 'selection': 'No dynamic outcome filtering'}
    write_new(Path(common.OUT) / f'collection-{split}-{uuid.uuid4().hex}.json', report)
    if failures:
        raise RuntimeError('Collection stopped at a hard failure; completed and incomplete evidence retained')


def audit_split(protocol, split):
    total = int(settings(protocol)['counts'][split])
    shards = discover_shards(split, protocol, verify=True)
    cursor = 0
    reset_ids, initial_hashes = [], []
    cells = {(lock, profile): 0 for lock in range(5) for profile in PROFILES}
    selected_shards = []
    audit_models = {}
    for shard in shards:
        if shard['start'] != cursor or shard['end'] - shard['start'] != shard['count']:
            raise RuntimeError(f'Noncontiguous or incorrect shard coverage: {shard["manifest_path"]}')
        records = common.read(shard['records_path'])
        if len(records) != shard['count']:
            raise RuntimeError('Metadata/array count mismatch')
        with np.load(shard['npz_path'], allow_pickle=False) as arrays:
            if arrays['states'].shape[1:] != (51, 14) or arrays['full_qpos'].shape[1:] != (51, 8) or arrays['full_qvel'].shape[1:] != (51, 8):
                raise RuntimeError('Unexpected full-state shape')
            if arrays['segment_actions'].shape[1:] != (5, 5):
                raise RuntimeError('Unexpected action shape')
            if np.max(abs(arrays['segment_actions'])) > settings(protocol)['action_limit']:
                raise RuntimeError('Action exceeds frozen bounds')
            for offset, record in enumerate(records):
                expected = identity_at(cursor + offset, total)
                actual = (int(arrays['locked_joint'][offset]), str(arrays['profile'][offset]), record['index'])
                if actual != expected or record['split'] != split or record['control_kind'] != 'random':
                    raise RuntimeError('Incorrect deterministic trajectory identity')
                if not record['reset_record']['passed'] or not record['diagnostics']['finite']:
                    raise RuntimeError('Initial-state or finite-state hard gate failed')
                if np.any(arrays['segment_actions'][offset, :, actual[0]] != 0.):
                    raise RuntimeError('Locked action must remain zero')
                expected_actions = action_segments(actual[0], actual[1], split, actual[2], 'random', protocol)
                if not np.array_equal(expected_actions, arrays['segment_actions'][offset]):
                    raise RuntimeError('Actions do not match the frozen per-identity RNG')
                if np.any(arrays['full_qvel'][offset, 0] != 0.):
                    raise RuntimeError('Initial velocity must remain zero')
                initial_sha = hashlib.sha256(np.concatenate((arrays['full_qpos'][offset, 0],
                    arrays['full_qvel'][offset, 0])).astype('<f8').tobytes()).hexdigest()
                if initial_sha != str(arrays['initial_state_sha256'][offset]) or initial_sha != record['initial_state_sha256']:
                    raise RuntimeError('Initial-state fingerprint mismatch')
                rid = str(arrays['reset_id'][offset])
                if rid != record['reset_record']['reset_id']:
                    raise RuntimeError('Reset id/metadata mismatch')
                if not np.array_equal(arrays['full_qpos'][offset, 0], record['reset_record']['qpos']):
                    raise RuntimeError('Initial full qpos disagrees with reset audit metadata')
                cell = actual[:2]
                if cell not in audit_models:
                    audit_models[cell] = common.make_reset(actual[0], actual[1], split, 0)[:2]
                model, initial = audit_models[cell]
                mujoco.mj_resetData(model, initial)
                initial.qpos[:] = arrays['full_qpos'][offset, 0]
                initial.qvel[:] = arrays['full_qvel'][offset, 0]
                activate_joint_lock(model, initial, checked.JOINTS[actual[0]], record['reset_record']['lock_angle_rad'])
                if not supported.inspect_reset(model, initial, actual[0])['passed']:
                    raise RuntimeError('Independent reconstructed initial geometry/support audit failed')
                if not np.array_equal(common.state14(model, initial), arrays['states'][offset, 0]):
                    raise RuntimeError('Initial learning projection does not match full state')
                reset_ids.append(rid)
                initial_hashes.append(initial_sha)
                cells[(actual[0], actual[1])] += 1
        cursor = shard['end']
        selected_shards.append({key: shard[key] for key in ('start', 'end', 'count', 'npz_path',
            'npz_sha256', 'records_path', 'records_sha256', 'manifest_path', 'manifest_sha256')})
    if cursor != total or len(set(reset_ids)) != total or len(set(initial_hashes)) != total:
        raise RuntimeError(f'Incomplete or duplicate split identities: {split}: {cursor}/{total}')
    if any(count != total // 20 for count in cells.values()):
        raise RuntimeError('Unbalanced lock/profile cells')
    return {'passed': True, 'count': cursor, 'shards': selected_shards,
        'cell_counts': {f'D{lock + 1}/{profile}': count for (lock, profile), count in cells.items()},
        'reset_ids': reset_ids, 'initial_state_sha256': initial_hashes}


def audit(protocol, split=None):
    require_physics_gate(protocol)
    splits = [split] if split else ['pool', 'validation', 'test']
    path = Path(common.OUT) / (f'data-audit-{split}.json' if split else 'data-audit.json')
    if path.exists():
        raise FileExistsError(f'Preserve previous audit: {path}')
    report = {'scope': 'Physics/state/identity integrity only; no learned-model test evaluation',
        'protocol_sha256': protocol_hash(protocol), 'source_sha256': source_hashes(),
        'splits': {}, 'files': {}, 'files_relative_to': str(Path(common.OUT).resolve()), 'passed': False}
    try:
        for name in splits:
            report['splits'][name] = audit_split(protocol, name)
        if not split:
            for field in ('reset_ids', 'initial_state_sha256'):
                sets = [set(report['splits'][name][field]) for name in splits]
                if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                    raise RuntimeError(f'Cross-split {field} overlap')
            dev = Path(common.OUT) / protocol.get('development_directory', 'data-development') / 'data.npz'
            if dev.exists():
                with np.load(dev, allow_pickle=False) as arrays:
                    dev_ids = set(map(str, arrays['reset_id']))
                    dev_initials = set(map(str, arrays['initial_state_sha256']))
                for name in splits:
                    if dev_ids & set(report['splits'][name]['reset_ids']) or dev_initials & set(report['splits'][name]['initial_state_sha256']):
                        raise RuntimeError('Development/formal initial identity overlap')
            report['cross_split_and_development_disjoint'] = True
        for value in report['splits'].values():
            for shard in value['shards']:
                for path_key, hash_key in (('npz_path', 'npz_sha256'), ('records_path', 'records_sha256'), ('manifest_path', 'manifest_sha256')):
                    relative = Path(shard[path_key]).relative_to(Path(common.OUT)).as_posix()
                    report['files'][relative] = shard[hash_key]
            value['reset_ids_sha256'] = digest_json(value.pop('reset_ids'))
            value['initial_fingerprints_sha256'] = digest_json(value.pop('initial_state_sha256'))
        report['passed'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    write_new(path, report)
    print(json.dumps({'audit': str(path), 'passed': report['passed']}, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Dataset audit failed; evidence preserved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('development', 'collect', 'audit'))
    parser.add_argument('--split', choices=('pool', 'validation', 'test'))
    args = parser.parse_args()
    protocol = common.load_protocol()
    if args.stage == 'development':
        development(protocol)
    elif args.stage == 'collect':
        collect(protocol, args.split)
    else:
        audit(protocol, args.split)


if __name__ == '__main__':
    main()
