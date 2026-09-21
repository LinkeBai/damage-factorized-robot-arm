"""Frozen, unit-separated prediction evaluation after all nine selections.

The reference is the independently rolled selected no-object-residual carrier
of the SAME fitting seed, under identical initial state, diagnosis and raw
commands. This is a reference-output diagnostic, not an isolation ablation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / 'src')]
from work.supported_core_compare.common import ROOT, OUT, PROTOCOL, load_protocol, read, sha, write
from work.supported_core_compare.train import load_selected, verify_selection_freeze
from robotarm.models.contact_geometry import pusher_reference_point

METHODS = ('ipwm', 'carrier', 'global')
SEEDS = (7, 17, 27)
HORIZONS = (10, 25, 50)
REFERENCE_DEFINITION = (
    'Selected carrier/no-object-residual model of the same fitting seed; '
    'standalone private recurrence from the same initial state, diagnosed lock '
    'angle and identical raw command sequence. This is not an independently '
    'trained robot model and not a full-state versus isolation ablation.'
)


def verify_implementation():
    protocol = load_protocol()
    implementation = read(OUT / 'implementation-frozen.json')
    if implementation.get('protocol_sha256') != sha(PROTOCOL):
        raise RuntimeError('Implementation freeze does not match the protocol')
    for manifest in (protocol.get('source_sha256', {}), implementation.get('source_sha256', {})):
        if not manifest:
            raise RuntimeError('Missing frozen source manifest')
        for relative, digest in manifest.items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f'Frozen source changed: {relative}')
    for name in ('evaluate.py', 'planning.py', 'train.py', 'data.py', 'common.py'):
        key = 'work/supported_core_compare/' + name
        if key not in implementation['source_sha256']:
            raise RuntimeError(f'Implementation freeze does not cover {name}')
    return implementation


def require_selection_freeze():
    """Never open test data or planning states before all selections are fixed."""
    verify_implementation()
    # The trainer verifies source, fitting-data and complete-record provenance.
    verify_selection_freeze()
    path = OUT / 'training-complete.json'
    frozen = read(path)
    if frozen.get('protocol_sha256', frozen.get('identity', {}).get('protocol_sha256')) != sha(PROTOCOL):
        raise RuntimeError('All-selection freeze does not match the protocol')
    expected = {f'{method}/seed{seed}' for method in METHODS for seed in SEEDS}
    if set(frozen.get('models', {})) != expected:
        raise RuntimeError('Exactly nine selected models are required before test access')
    for key, digest in frozen['models'].items():
        folder = OUT / 'training' / key
        complete = read(folder / 'complete.json')
        actual = sha(folder / 'model.pt')
        if actual != digest or complete.get('model_sha256') != actual:
            raise RuntimeError(f'Selected checkpoint changed after freeze: {key}')
    return frozen


def checked_test_data():
    """Load only committed, hash-verified independent test shards."""
    from work.supported_core_compare.data import discover_shards
    records = discover_shards('test', protocol=load_protocol(), verify=True)
    if not records:
        raise RuntimeError('No committed test data')
    fields = ('states', 'segment_actions', 'locked_joint', 'profile', 'reset_id',
              'initial_state_sha256')
    arrays = {field: [] for field in fields}
    sources = []
    for record in records:
        path = Path(record['npz_path'])
        with np.load(path, allow_pickle=False) as archive:
            for field in fields:
                arrays[field].append(archive[field])
        sources.append({'path': str(path), 'sha256': sha(path),
                        'manifest_sha256': sha(record['manifest_path'])})
    arrays = {field: np.concatenate(parts) for field, parts in arrays.items()}
    count = len(arrays['states'])
    if count != 6000 or arrays['states'].shape != (count, 51, 14):
        raise RuntimeError(f'Expected 6000 complete 50-step test trajectories, got {count}')
    if arrays['segment_actions'].shape != (count, 5, 5):
        raise RuntimeError('Incorrect test command shape')
    if len(np.unique(arrays['reset_id'])) != count:
        raise RuntimeError('Test identities are not unique')
    if not np.isfinite(arrays['states']).all():
        raise RuntimeError('Nonfinite ground-truth state; test rows must not be filtered')
    if not np.isin(arrays['locked_joint'], np.arange(5)).all():
        raise RuntimeError('Invalid lock index')
    return arrays, sources


def metric_arrays(prediction, truth, reference, locks, angles):
    """Per-trajectory metrics; never add quantities with different units."""
    pred = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    free = 1. - np.eye(5)[locks]
    error = pred - actual
    ref_error = pred[:, :10] - ref[:, :10]
    with torch.no_grad():
        pusher_pred = pusher_reference_point(torch.from_numpy(pred[:, :5])).numpy()[:, :2]
        pusher_truth = pusher_reference_point(torch.from_numpy(actual[:, :5])).numpy()[:, :2]
        pusher_ref = pusher_reference_point(torch.from_numpy(ref[:, :5])).numpy()[:, :2]
    row = np.arange(len(pred))
    return {
        'object_xy_squared_m2': error[:, 10:12] ** 2,
        'object_velocity_squared_m2_s2': error[:, 12:14] ** 2,
        'joint_q_squared_rad2': error[:, :5] ** 2,
        'joint_velocity_squared_rad2_s2': error[:, 5:10] ** 2,
        'free_q_mean_squared_rad2': (error[:, :5] ** 2 * free).sum(1) / 4.,
        'free_velocity_mean_squared_rad2_s2': (error[:, 5:10] ** 2 * free).sum(1) / 4.,
        'pusher_xy_squared_m2': (pusher_pred - pusher_truth) ** 2,
        'lock_position_abs_rad': abs(pred[row, locks] - angles),
        'lock_velocity_abs_rad_s': abs(pred[row, locks + 5]),
        'reference_q_abs_rad': abs(ref_error[:, :5]),
        'reference_velocity_abs_rad_s': abs(ref_error[:, 5:10]),
        'reference_pusher_xy_abs_m': abs(pusher_pred - pusher_ref),
        'predicted_state14': pred,
        'reference_state14': ref,
        'truth_state14': actual,
    }


def summarize_rows(rows):
    """RMSE is per-coordinate unless the name explicitly says maximum."""
    result = {}
    for horizon in HORIZONS:
        prefix = f'h{horizon}_'
        result[str(horizon)] = {
            'object_position_per_coordinate_rmse_mm': 1000. * float(np.sqrt(rows[prefix + 'object_xy_squared_m2'].mean())),
            'object_velocity_per_coordinate_rmse_m_s': float(np.sqrt(rows[prefix + 'object_velocity_squared_m2_s2'].mean())),
            'free_joint_position_rmse_rad': float(np.sqrt(rows[prefix + 'free_q_mean_squared_rad2'].mean())),
            'free_joint_velocity_rmse_rad_s': float(np.sqrt(rows[prefix + 'free_velocity_mean_squared_rad2_s2'].mean())),
            'fk_pusher_position_per_coordinate_rmse_mm': 1000. * float(np.sqrt(rows[prefix + 'pusher_xy_squared_m2'].mean())),
            'lock_position_max_abs_rad': float(rows[prefix + 'lock_position_abs_rad'].max()),
            'lock_velocity_max_abs_rad_s': float(rows[prefix + 'lock_velocity_abs_rad_s'].max()),
            'reference_q_max_abs_rad': float(rows[prefix + 'reference_q_abs_rad'].max()),
            'reference_velocity_max_abs_rad_s': float(rows[prefix + 'reference_velocity_abs_rad_s'].max()),
            'reference_pusher_xy_max_abs_mm': 1000. * float(rows[prefix + 'reference_pusher_xy_abs_m'].max()),
        }
    return result


def completed_result(folder, identity):
    if not folder.exists():
        return False
    complete = read(folder / 'complete.json')
    if complete.get('identity') != identity or complete.get('rows_sha256') != sha(folder / 'rows.npz'):
        raise RuntimeError(f'Existing result has different inputs or changed rows: {folder}')
    return True


@torch.no_grad()
def evaluate_one(method, seed, model, model_hash, reference, reference_hash,
                 arrays, sources, *, device, batch_size=512):
    frozen = require_selection_freeze()
    identity = {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(__file__),
                'implementation_freeze_sha256': sha(OUT / 'implementation-frozen.json'),
                'selection_freeze_sha256': sha(OUT / 'training-complete.json'),
                'model_sha256': model_hash, 'reference_model_sha256': reference_hash,
                'test_sources': sources}
    folder = OUT / 'prediction' / method / f'seed{seed}'
    if completed_result(folder, identity):
        return read(folder / 'complete.json')
    if frozen['models'][f'{method}/seed{seed}'] != model_hash:
        raise RuntimeError('Loaded model does not match frozen selection')
    rows = {key: arrays[key].copy() for key in ('reset_id', 'initial_state_sha256', 'locked_joint', 'profile')}
    rows['free_joint_mask'] = 1 - np.eye(5, dtype=np.int8)[arrays['locked_joint']]
    results = {h: [] for h in HORIZONS}
    reference_max_q, reference_max_v = [], []
    lock_max_q, lock_max_v = [], []
    began = time.perf_counter()
    for start in range(0, len(arrays['states']), batch_size):
        stop = min(start + batch_size, len(arrays['states']))
        truth = arrays['states'][start:stop]
        locks = arrays['locked_joint'][start:stop].astype(np.int64)
        mask = torch.tensor(np.eye(5)[locks], dtype=torch.float32, device=device)
        initial = torch.as_tensor(truth[:, 0], dtype=torch.float32, device=device)
        lock_angles = initial[:, :5] * mask
        angles_np = truth[np.arange(len(truth)), 0, locks]
        commands = torch.as_tensor(arrays['segment_actions'][start:stop], dtype=torch.float32, device=device)
        x, xr, hidden, hr = initial, initial.clone(), None, None
        max_q = np.zeros(len(truth)); max_v = max_q.copy()
        max_lq = max_q.copy(); max_lv = max_q.copy()
        for step in range(50):
            u = commands[:, step // 10]
            x, hidden = model.step(x, u, mask, lock_angles, hidden)
            xr, hr = reference.step(xr, u, mask, lock_angles, hr)
            if not torch.isfinite(x).all() or not torch.isfinite(xr).all():
                raise RuntimeError(f'Nonfinite prediction: {method}/seed{seed}, rows {start}:{stop}, step {step + 1}')
            p, r = x.cpu().numpy(), xr.cpu().numpy()
            max_q = np.maximum(max_q, abs(p[:, :5] - r[:, :5]).max(1))
            max_v = np.maximum(max_v, abs(p[:, 5:10] - r[:, 5:10]).max(1))
            max_lq = np.maximum(max_lq, abs(p[np.arange(len(p)), locks] - angles_np))
            max_lv = np.maximum(max_lv, abs(p[np.arange(len(p)), locks + 5]))
            if step + 1 in results:
                results[step + 1].append(metric_arrays(p, truth[:, step + 1], r, locks, angles_np))
        reference_max_q.append(max_q); reference_max_v.append(max_v)
        lock_max_q.append(max_lq); lock_max_v.append(max_lv)
        print(f'prediction {method} seed{seed}: {stop}/{len(arrays["states"])}', flush=True)
    for horizon, batches in results.items():
        for key in batches[0]:
            rows[f'h{horizon}_{key}'] = np.concatenate([batch[key] for batch in batches])
    rows.update(reference_q_all_steps_max_abs_rad=np.concatenate(reference_max_q),
                reference_velocity_all_steps_max_abs_rad_s=np.concatenate(reference_max_v),
                lock_position_all_steps_max_abs_rad=np.concatenate(lock_max_q),
                lock_velocity_all_steps_max_abs_rad_s=np.concatenate(lock_max_v))
    temporary = folder.with_name(folder.name + f'.incomplete-{os.getpid()}-{uuid.uuid4().hex}')
    temporary.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(temporary / 'rows.npz', **rows)
    report = {'method': method, 'seed': seed, 'identity': identity,
              'independent_test_trajectories': len(arrays['states']),
              'horizons': list(HORIZONS), 'reference_definition': REFERENCE_DEFINITION,
              'pusher_definition': 'Fixed analytic pusher_reference_point applied to predicted versus measured joint positions; not an independently sensed tip position.',
              'position_rmse_definition': 'sqrt(mean across trajectories and the two coordinate squared errors)); meters converted to mm only in summary.',
              'state_projection': '14-D learning input excludes recorded block_z and block_vz; full dynamics remain in test data.',
              'claim_boundary': 'Three adaptation models; no attribution to isolation or candidate safety gates.',
              'summary': summarize_rows(rows), 'seconds': time.perf_counter() - began,
              'rows_sha256': sha(temporary / 'rows.npz')}
    write(temporary / 'complete.json', report)
    temporary.rename(folder)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.all and (args.method is None or args.seed is None):
        parser.error('Use --all or both --method and --seed')
    if args.all and (args.method is not None or args.seed is not None):
        parser.error('--all cannot be combined with a selected method/seed')
    torch.set_num_threads(2)
    require_selection_freeze()
    arrays, sources = checked_test_data()
    for seed in SEEDS if args.all else (args.seed,):
        reference, reference_hash = load_selected('carrier', seed, args.device)
        for method in METHODS if args.all else (args.method,):
            model, model_hash = load_selected(method, seed, args.device)
            evaluate_one(method, seed, model, model_hash, reference, reference_hash,
                         arrays, sources, device=args.device)
            del model
        del reference


if __name__ == '__main__':
    main()
