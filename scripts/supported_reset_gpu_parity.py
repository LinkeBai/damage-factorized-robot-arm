"""Bounded full-state development parity for the supported reset revision.

This audits the additional block z/vz state and capture/reset invariants. It
does not change the original parity tolerances or establish model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mujoco
import numpy as np
from robotarm.envs import supported_push_reset as supported
from scripts.validate_ipwm_checked_reset import clone

TOLERANCES = {'object_position_m': .001, 'robot_position_rad': .002,
    'object_velocity_mps': .01, 'robot_velocity_rad_s': .02,
    'block_z_position_m': .001, 'block_z_velocity_m_s': .01}


def gpu_parity(lock, profile, count=4):
    """Compare 50 identical CPU/GPU steps, including all eight DOFs."""
    import mujoco_warp as mjw
    import warp as wp
    if not 1 <= count <= 4:
        raise ValueError('Development parity is bounded to four worlds per cell')
    m = supported.make_model(lock, profile)
    sources = [supported.sample_reset(m, lock, profile, 'development', i)[0]
               for i in range(count)]
    eq = m.equality('fault_lock_' + supported.JOINTS[lock]).id
    aq = [int(m.joint(n).qposadr[0]) for n in supported.JOINTS]
    av = [int(m.joint(n).dofadr[0]) for n in supported.JOINTS]
    bq = [int(m.joint(n).qposadr[0]) for n in ('block_x', 'block_y')]
    bv = [int(m.joint(n).dofadr[0]) for n in ('block_x', 'block_y')]
    zq = int(m.joint('block_z').qposadr[0])
    zv = int(m.joint('block_z').dofadr[0])
    assert sorted(aq + bq + [zq]) == list(range(m.nq))
    assert sorted(av + bv + [zv]) == list(range(m.nv))
    qp = np.stack([d.qpos for d in sources])
    qv = np.stack([d.qvel for d in sources])
    times = np.array([d.time for d in sources])
    expected_active = np.stack([d.eq_active for d in sources])
    expected_lock_mask = np.zeros_like(expected_active)
    expected_lock_mask[:, eq] = True
    assert np.array_equal(expected_active, expected_lock_mask)
    assert np.all(qv == 0.) and np.all(times == 0.)
    angles = qp[:, aq[lock]].copy()
    m.eq_active0[:] = expected_active[0]
    wm = mjw.put_model(m, batch_sizes={'eq_data': count})
    eqdata = wm.eq_data.numpy()
    assert eqdata.shape[0] == count
    eqdata[:, eq, 0] = angles
    wm.eq_data.assign(eqdata)
    empty = mujoco.MjData(m)
    wp.init()

    def fresh_gpu_data():
        data = mjw.put_data(m, empty, nworld=count)
        data.qpos.assign(qp.astype(np.float32))
        data.qvel.assign(qv.astype(np.float32))
        data.time.assign(times.astype(data.time.numpy().dtype))
        if hasattr(data, 'eq_active'):
            data.eq_active.assign(expected_active.astype(data.eq_active.numpy().dtype))
        return data

    # Warm-up executes only on disposable data, then capture uses fresh state.
    wd = fresh_gpu_data()
    mjw.step(wm, wd)
    wp.synchronize()
    wd = fresh_gpu_data()
    with wp.ScopedCapture() as cap:
        mjw.step(wm, wd)
    post_qp, post_qv, post_time = wd.qpos.numpy(), wd.qvel.numpy(), wd.time.numpy()
    post_eqdata = wm.eq_data.numpy()
    checks = {
        'qpos_per_world_exact': bool(np.array_equal(post_qp, qp.astype(post_qp.dtype))),
        'qvel_per_world_exact': bool(np.array_equal(post_qv, qv.astype(post_qv.dtype))),
        'time_per_world_exact': bool(np.array_equal(post_time, times.astype(post_time.dtype))),
        'batched_eq_data_exact': bool(np.array_equal(post_eqdata, eqdata)),
        'lock_angles_per_world_exact': bool(np.array_equal(post_eqdata[:, eq, 0],
                                                         angles.astype(post_eqdata.dtype))),
        'eq_active_available': hasattr(wd, 'eq_active'),
    }
    if checks['eq_active_available']:
        actual_active = wd.eq_active.numpy()
        checks['eq_active_per_world_exact'] = bool(np.array_equal(
            actual_active, expected_active.astype(actual_active.dtype)))
    else:
        checks['eq_active_per_world_exact'] = None
        checks['fallback_model_eq_active0_exact'] = bool(np.array_equal(
            m.eq_active0, expected_active[0]))
    checks['passed'] = all(v for k, v in checks.items()
                           if k != 'eq_active_available' and v is not None)
    if not checks['passed']:
        raise AssertionError(f'Postcapture reset invariant failed: {checks}')
    rng = np.random.default_rng(np.random.SeedSequence(
        [9132026, 808, lock, supported.PROFILES.index(profile)]))
    actions = rng.uniform(-.2, .2, (count, 5, 5)).astype(np.float32)
    actions[:, :, lock] = 0
    cpu = [clone(m, d) for d in sources]
    maxima = {name: 0. for name in TOLERANCES}
    cpu_eq_assertions = 0
    for t in range(50):
        if t % 10 == 0:
            wd.ctrl.assign(actions[:, t // 10])
        wp.capture_launch(cap.graph)
        gq, gv = wd.qpos.numpy(), wd.qvel.numpy()
        for i, d in enumerate(cpu):
            # CPU worlds share MjModel: restore each world's equality target
            # immediately before its step, matching the batched GPU eq_data.
            m.eq_data[eq, 0] = angles[i]
            assert m.eq_data[eq, 0] == angles[i]
            assert np.array_equal(d.eq_active, expected_active[i])
            cpu_eq_assertions += 1
            d.ctrl[:] = actions[i, t // 10]
            mujoco.mj_step(m, d)
            for name, delta in (
                ('object_position_m', abs(gq[i, bq] - d.qpos[bq])),
                ('robot_position_rad', abs(gq[i, aq] - d.qpos[aq])),
                ('object_velocity_mps', abs(gv[i, bv] - d.qvel[bv])),
                ('robot_velocity_rad_s', abs(gv[i, av] - d.qvel[av])),
                ('block_z_position_m', abs(gq[i, zq] - d.qpos[zq])),
                ('block_z_velocity_m_s', abs(gv[i, zv] - d.qvel[zv]))):
                value = float(np.max(delta))
                if not np.isfinite(value):
                    raise ValueError('Nonfinite CPU/GPU difference')
                maxima[name] = max(maxima[name], value)
    return {'lock': lock + 1, 'profile': profile, 'count': count,
        'maxima': maxima, 'tolerances': dict(TOLERANCES),
        'capture_checks': checks, 'capture_time_s': post_time.tolist(),
        'capture_lock_angles_rad': post_eqdata[:, eq, 0].tolist(),
        'expected_lock_angles_rad': angles.tolist(),
        'cpu_per_world_eq_assertions': cpu_eq_assertions,
        'failed_metrics': [name for name, limit in TOLERANCES.items()
                           if maxima[name] >= limit],
        'passed': checks['passed'] and all(maxima[n] < lim for n, lim in TOLERANCES.items())}


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (supported.XML, ROOT / 'src/robotarm/envs/checked_push_reset.py',
            ROOT / 'src/robotarm/envs/supported_push_reset.py',
            ROOT / 'src/robotarm/envs/constraint_lock.py',
            ROOT / 'scripts/validate_ipwm_checked_reset.py', Path(__file__))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'runs/ipwm_supported_reset_development/gpu-full-state-parity.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior development evidence: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    rows = []
    for lock in range(5):
        for profile in supported.PROFILES:
            row = gpu_parity(lock, profile)
            rows.append(row)
            print(json.dumps(row), flush=True)
    after = source_hashes()
    report = {
        'scope': 'Development initialization full-state CPU/GPU parity; no model performance results',
        'model_revision': supported.VERSION, 'source_sha256': before,
        'source_unchanged_during_run': before == after,
        'mujoco_version': mujoco.__version__,
        'mujoco_warp_version': importlib.metadata.version('mujoco-warp'),
        'warp_version': importlib.metadata.version('warp-lang'),
        'paired_worlds': sum(r['count'] for r in rows), 'steps_per_world': 50,
        'worlds_per_cell': 4, 'cells': len(rows),
        'state_audit': 'All 8 qpos and 8 qvel entries by named joints, including block_z/vz',
        'rows': rows, 'passed_cells': sum(r['passed'] for r in rows),
        'capture_checks_passed': all(r['capture_checks']['passed'] for r in rows),
        'maxima': {name: max(r['maxima'][name] for r in rows) for name in TOLERANCES},
        'tolerances': TOLERANCES,
        'passed': before == after and all(r['passed'] for r in rows),
        'production_batch_validated': False, 'model_performance_validated': False,
        'global_dynamic_accuracy_validated': False,
        'scale_run_authorized_by_this_report': False,
        'prior_xy_arm_evidence': 'gpu-parity.json retained, including its D2 weak_motor failure',
    }
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Development full-state CPU/GPU parity failed; preserved all results')


if __name__ == '__main__':
    main()
