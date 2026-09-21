"""Independent reset trajectories for the registered scale study.

Solver-native locks, raw motor commands, and SI state coordinates match the
fresh candidate protocol. This is a new simulation dataset, not real evidence.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from scripts.generate_primary_sequence_candidates import JOINTS, CONTACT_QPOS
from robotarm.envs.constraint_lock import model_with_inactive_joint_locks

OUT = ROOT / 'runs/ipwm_scale_goal_20260911'
PROFILES = ('nominal', 'high_damping', 'weak_motor', 'mixed')
SPLITS = {'pool': (10000, 101), 'validation': (400, 202), 'test': (1200, 303)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_model(lock, profile):
    m = model_with_inactive_joint_locks(ROOT / 'sim/assets/arm_push.xml', JOINTS)
    ids = [int(m.joint(j).dofadr[0]) for j in JOINTS]
    if profile in ('high_damping', 'mixed'):
        m.dof_damping[ids] *= 2.
    if profile in ('weak_motor', 'mixed'):
        m.actuator_gear[:, 0] *= .7
    eq = m.equality('fault_lock_' + JOINTS[lock]).id
    m.eq_active0[:] = 0
    m.eq_active0[eq] = 1
    m.eq_data[eq, 0] = CONTACT_QPOS[lock]
    return m, eq

def samples(m, split, lock, profile_id, start, n):
    # A unique generator per reset makes sampling independent of batch sizes.
    qp = np.tile(m.qpos0, (n, 1)).astype(np.float32)
    qv = np.zeros((n, m.nv), dtype=np.float32)
    actions = np.empty((n, 5, 5), dtype=np.float32)
    armq = [int(m.joint(j).qposadr[0]) for j in JOINTS]
    armv = [int(m.joint(j).dofadr[0]) for j in JOINTS]
    blockq = [int(m.joint(j).qposadr[0]) for j in ('block_x', 'block_y')]
    ids = []
    for i in range(n):
        identity = (SPLITS[split][1], lock, profile_id, start + i)
        rng = np.random.default_rng(np.random.SeedSequence([9112026, *identity]))
        qp[i, armq] = CONTACT_QPOS + rng.normal(0, .015, 5)
        qv[i, armv] = rng.normal(0, .01, 5)
        qv[i, armv[lock]] = 0
        qp[i, blockq] = rng.uniform(-.004, .004, 2)
        actions[i] = rng.uniform(-.8, .8, (5, 5))
        actions[i, :, lock] = 0
        ids.append('%s-D%d-P%d-%06d' % (split, lock + 1, profile_id, start + i))
    return qp, qv, actions, np.asarray(ids), armq, armv, blockq

def simulate(lock, profile, split, start, n, cpu_check=0):
    m, eq = make_model(lock, profile)
    qp, qv, actions, ids, armq, armv, blockq = samples(m, split, lock, PROFILES.index(profile), start, n)
    blockv = [int(m.joint(j).dofadr[0]) for j in ('block_x', 'block_y')]
    origin = m.body('block').pos[:2]
    def state(q, v):
        return np.concatenate((q[:, armq], v[:, armv], q[:, blockq] + origin, v[:, blockv]), axis=1)
    wm = mjw.put_model(m, batch_sizes={'eq_data': n})
    eqdata = wm.eq_data.numpy()
    eqdata[:, eq, 0] = qp[:, armq[lock]]
    wm.eq_data.assign(eqdata)
    d = mujoco.MjData(m)
    wd = mjw.put_data(m, d, nworld=n)
    wd.qpos.assign(qp)
    wd.qvel.assign(qv)
    states = [state(qp, qv)]
    # Compile all kernels before capture, then restore the exact reset.
    mjw.step(wm, wd)
    wp.synchronize()
    wd.qpos.assign(qp)
    wd.qvel.assign(qv)
    wd.qacc_warmstart.zero_()
    wd.time.zero_()
    with wp.ScopedCapture() as capture:
        mjw.step(wm, wd)
    began = time.perf_counter()
    for t in range(50):
        if t % 10 == 0:
            wd.ctrl.assign(actions[:, t // 10])
        wp.capture_launch(capture.graph)
        states.append(state(wd.qpos.numpy(), wd.qvel.numpy()))
    states = np.asarray(states, dtype=np.float32).transpose(1, 0, 2)
    if not np.isfinite(states).all():
        raise ValueError('Nonfinite physics')
    checks = []
    for i in range(min(cpu_check, n)):
        m.eq_data[eq, 0] = qp[i, armq[lock]]
        cd = mujoco.MjData(m)
        cd.qpos[:] = qp[i]
        cd.qvel[:] = qv[i]
        mujoco.mj_forward(m, cd)
        expected = [state(qp[i:i+1], qv[i:i+1])[0]]
        for t in range(50):
            cd.ctrl[:] = actions[i, t // 10]
            mujoco.mj_step(m, cd)
            expected.append(state(cd.qpos[None], cd.qvel[None])[0])
        expected = np.asarray(expected)
        err = np.linalg.norm(expected[:, 10:12] - states[i, :, 10:12], axis=-1)
        checks.append({'reset_id': str(ids[i]), 'max_object_xy_difference_m': float(err.max()),
                       'max_robot_q_difference_rad': float(np.abs(expected[:, :5] - states[i, :, :5]).max())})
    return dict(states=states, segment_actions=actions, reset_id=ids,
                locked_joint=np.full(n, lock, dtype=np.int8),
                profile=np.full(n, PROFILES.index(profile), dtype=np.int8)), dict(
                    seconds=time.perf_counter()-began, cpu_checks=checks,
                    max_lock_position_deviation_rad=float(np.abs(states[:, :, lock] - states[:, :1, lock]).max()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--split', choices=list(SPLITS))
    p.add_argument('--lock', type=int, choices=range(1, 6))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wp.init()
    if args.validate:
        rows = []
        for lock in range(5):
            for profile in PROFILES:
                _, check = simulate(lock, profile, 'validation', 900000, 32, cpu_check=8)
                rows.append(dict(lock=lock+1, profile=profile, **check))
                print('validated', lock+1, profile, flush=True)
        passed = all(c['max_object_xy_difference_m'] < .001 and c['max_robot_q_difference_rad'] < .002
                     for row in rows for c in row['cpu_checks'])
        result = dict(pilot_only=True, passed=passed, object_threshold_m=.001, robot_threshold_rad=.002,
                      script_sha256=sha(__file__), xml_sha256=sha(ROOT/'sim/assets/arm_push.xml'), rows=rows)
        (OUT/'simulator-validation.json').write_text(json.dumps(result, indent=2))
        if not passed:
            raise ValueError('GPU parity threshold failed')
        return
    check = json.loads((OUT/'simulator-validation.json').read_text())
    assert check['passed'] and check['script_sha256'] == sha(__file__)
    frozen = json.loads((OUT/'data-protocol.json').read_text())
    assert frozen['script_sha256'] == sha(__file__)
    assert frozen['xml_sha256'] == sha(ROOT/'sim/assets/arm_push.xml')
    count = SPLITS[args.split][0] // len(PROFILES)
    folder = OUT/'data'/args.split/f'D{args.lock}'
    folder.mkdir(parents=True, exist_ok=True)
    manifests = []
    for pi, profile in enumerate(PROFILES):
        for start in range(0, count, 500):
            path = folder/f'{profile}-{start:05d}.npz'
            meta = path.with_suffix('.json')
            if path.exists() and meta.exists():
                record = json.loads(meta.read_text())
                assert record['sha256'] == sha(path)
                manifests.append(record)
                continue
            arrays, timing = simulate(args.lock-1, profile, args.split, start, min(500, count-start))
            np.savez_compressed(path, **arrays)
            record = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), count=len(arrays['reset_id']),
                          split=args.split, lock=args.lock, profile=profile, **timing)
            meta.write_text(json.dumps(record, indent=2))
            manifests.append(record)
            print(args.split, args.lock, profile, start, record['count'], flush=True)
    (folder/'manifest.json').write_text(json.dumps(dict(count=sum(r['count'] for r in manifests), shards=manifests), indent=2))

if __name__ == '__main__':
    main()
