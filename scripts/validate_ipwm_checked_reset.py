"""Bounded development checks; never train or publish performance results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mujoco
import numpy as np
from robotarm.envs.checked_push_reset import (
    ARM_GEOMS, JOINTS, PROFILES, XML, ResetSpec, geometry, inspect_reset,
    make_model, planning_reset, sample_reset,
)


def clone(m, source):
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, source)
    return d


def state(m, d):
    return np.concatenate((d.qpos[[int(m.joint(n).qposadr[0]) for n in JOINTS]],
        d.qvel[[int(m.joint(n).dofadr[0]) for n in JOINTS]],
        d.body('block').xpos[:2],
        d.qvel[[int(m.joint(n).dofadr[0]) for n in ('block_x', 'block_y')]]))


def dynamic_probe(m, initial, lock, kind):
    d = clone(m, initial)
    initial_state = state(m, d)
    # Fixed development controls, not selected for task success.
    ctrl = np.zeros(5)
    free = [j for j in range(5) if j != lock]
    if kind != 'zero':
        ctrl[free[0]] = .2 if kind == 'positive' else -.2
    rows = []
    max_delta_without_contact = 0.
    ever_object_contact = False
    for step in range(51):
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        geo = geometry(m, d)
        contacts = []
        for ci, c in enumerate(d.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, force)
            a, b = m.geom(c.geom1).name, m.geom(c.geom2).name
            contacts.append({'geom1': a, 'geom2': b, 'distance_m': float(c.dist),
                             'normal_force_N': float(force[0])})
            if 'block_geom' in (a, b) and ({a, b} & {'tool_geom', 'pusher_geom'}) and force[0] > 1e-9:
                ever_object_contact = True
        s = state(m, d)
        if not ever_object_contact:
            max_delta_without_contact = max(max_delta_without_contact, float(np.linalg.norm(s[10:12]-initial_state[10:12])))
        armv = [int(m.joint(n).dofadr[0]) for n in JOINTS]
        expected_torque = np.clip(ctrl, m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]) * m.actuator_gear[:, 0]
        rows.append({'step': step, 'time_s': float(d.time), 'state': s.tolist(), 'ctrl': ctrl.tolist(),
            'finite': bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'minimum_arm_table_m': min(v for n, v in geo['arm_table_m'].items() if n != 'base_geom'),
            'minimum_arm_block_m': min(geo['arm_block_m'].values()),
            'minimum_joint_margin_rad': min(geo['joint_margins_rad'].values()),
            'lock_position_drift_rad': float(abs(s[lock]-initial_state[lock])),
            'lock_speed_rad_s': float(abs(s[5+lock])),
            'actuator_torque_error_Nm': float(np.max(abs(d.qfrc_actuator[armv]-expected_torque))),
            'contacts': contacts})
        if step < 50:
            mujoco.mj_step(m, d)
    return {'control_kind': kind, 'rows': rows,
        'object_displacement_m': float(np.linalg.norm(state(m, d)[10:12]-initial_state[10:12])),
        'max_object_displacement_without_prior_contact_m': max_delta_without_contact,
        'max_lock_drift_rad': max(r['lock_position_drift_rad'] for r in rows),
        'minimum_arm_table_m': min(r['minimum_arm_table_m'] for r in rows),
        'minimum_arm_block_m': min(r['minimum_arm_block_m'] for r in rows),
        'minimum_joint_margin_rad': min(r['minimum_joint_margin_rad'] for r in rows),
        'max_actuator_torque_error_Nm': max(r['actuator_torque_error_Nm'] for r in rows),
        'finite': all(r['finite'] for r in rows)}


def gpu_parity(lock, profile, count=4):
    import mujoco_warp as mjw
    import warp as wp
    m = make_model(lock, profile)
    sources = [sample_reset(m, lock, profile, 'development', i)[0] for i in range(count)]
    eq = m.equality('fault_lock_' + JOINTS[lock]).id
    aq = [int(m.joint(n).qposadr[0]) for n in JOINTS]
    av = [int(m.joint(n).dofadr[0]) for n in JOINTS]
    bq = [int(m.joint(n).qposadr[0]) for n in ('block_x', 'block_y')]
    bv = [int(m.joint(n).dofadr[0]) for n in ('block_x', 'block_y')]
    qp, qv = np.stack([d.qpos for d in sources]), np.stack([d.qvel for d in sources])
    angles = qp[:, aq[lock]].copy()
    m.eq_active0[:] = False; m.eq_active0[eq] = True
    wm = mjw.put_model(m, batch_sizes={'eq_data': count})
    eqdata = wm.eq_data.numpy(); eqdata[:, eq, 0] = angles; wm.eq_data.assign(eqdata)
    empty = mujoco.MjData(m)
    wd = mjw.put_data(m, empty, nworld=count)
    wp.init()
    # Warm-up and graph compilation use disposable data; clean data is recreated.
    wd.qpos.assign(qp.astype(np.float32)); wd.qvel.assign(qv.astype(np.float32))
    mjw.step(wm, wd); wp.synchronize()
    wd = mjw.put_data(m, empty, nworld=count)
    wd.qpos.assign(qp.astype(np.float32)); wd.qvel.assign(qv.astype(np.float32))
    with wp.ScopedCapture() as cap:
        mjw.step(wm, wd)
    # Graph capture does not execute the graph. These checks defend the reset.
    assert np.array_equal(wd.qpos.numpy(), qp.astype(np.float32))
    rng = np.random.default_rng(np.random.SeedSequence([9132026, 808, lock, PROFILES.index(profile)]))
    actions = rng.uniform(-.2, .2, (count, 5, 5)).astype(np.float32); actions[:, :, lock] = 0
    cpu = [clone(m, d) for d in sources]
    maxima = {'object_position_m': 0., 'robot_position_rad': 0.,
              'object_velocity_mps': 0., 'robot_velocity_rad_s': 0.}
    for t in range(50):
        if t % 10 == 0:
            wd.ctrl.assign(actions[:, t // 10])
        wp.capture_launch(cap.graph)
        gq, gv = wd.qpos.numpy(), wd.qvel.numpy()
        for i, d in enumerate(cpu):
            m.eq_data[eq, 0] = angles[i]
            d.ctrl[:] = actions[i, t // 10]
            mujoco.mj_step(m, d)
            for name, delta in (
                ('object_position_m', abs(gq[i, bq]-d.qpos[bq])),
                ('robot_position_rad', abs(gq[i, aq]-d.qpos[aq])),
                ('object_velocity_mps', abs(gv[i, bv]-d.qvel[bv])),
                ('robot_velocity_rad_s', abs(gv[i, av]-d.qvel[av]))):
                value = float(np.max(delta))
                if not np.isfinite(value):
                    raise ValueError('Nonfinite CPU/GPU difference')
                maxima[name] = max(maxima[name], value)
    tolerances = {'object_position_m': .001, 'robot_position_rad': .002,
                  'object_velocity_mps': .01, 'robot_velocity_rad_s': .02}
    return {'lock':lock+1, 'profile':profile, 'count': count,
            'maxima':maxima, 'tolerances':tolerances,
            'passed':all(maxima[n] < lim for n, lim in tolerances.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'runs/ipwm_checked_reset_development')
    parser.add_argument('--per-cell', type=int, default=60)
    parser.add_argument('--gpu', action='store_true')
    a = parser.parse_args()
    if not 1 <= a.per_cell <= 60:
        raise ValueError('This development script is bounded to at most 1200 resets')
    a.output.mkdir(parents=True, exist_ok=True)
    result_path = a.output / ('gpu-parity.json' if a.gpu else 'validation.json')
    if result_path.exists():
        raise FileExistsError(f'Preserve the prior development result: {result_path}')
    if a.gpu:
        rows = []
        for lock in range(5):
            for profile in PROFILES:
                row = gpu_parity(lock, profile); rows.append(row)
                print(json.dumps(row), flush=True)
        result_path.write_text(json.dumps({'rows':rows, 'passed':all(r['passed'] for r in rows),
            'production_batch_validated':False, 'scope':'4-world development parity, not a production release'},indent=2),encoding='utf-8')
        if not all(r['passed'] for r in rows):
            raise RuntimeError('Development CPU/GPU parity failed')
        return
    resets, probes, planning = [], [], []
    for lock in range(5):
        for pi, profile in enumerate(PROFILES):
            m = make_model(lock, profile)
            for i in range(a.per_cell):
                d, record = sample_reset(m, lock, profile, 'development', i)
                assert inspect_reset(m, d, lock)['passed']
                resets.append(record)
                if i == 0:
                    for kind in ('zero', 'positive', 'negative'):
                        probes.append({'lock':lock+1, 'profile':profile,
                            'reset_id':record['reset_id'], **dynamic_probe(m, d, lock, kind)})
            for band in (0, 1):
                pm, pd, goal, record = planning_reset(lock, profile, band, 0)
                assert inspect_reset(pm, pd, lock)['passed']
                planning.append({'goal':goal.tolist(), **record})
            print(f'geometry checked D{lock+1} {profile}', flush=True)
    (a.output/'reset-records.json').write_text(json.dumps(resets,indent=2),encoding='utf-8')
    (a.output/'dynamic-probes.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
    (a.output/'planning-resets.json').write_text(json.dumps(planning,indent=2),encoding='utf-8')
    report = {'scope':'Initialization repair and bounded original-dynamics diagnostics; no training, model scores, or task performance claims',
        'mujoco_version':mujoco.__version__, 'spec':asdict(ResetSpec()),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            XML, ROOT/'src/robotarm/envs/checked_push_reset.py',
            ROOT/'src/robotarm/envs/constraint_lock.py', Path(__file__))},
        'geometric_resets_checked':len(resets), 'geometry_passed':all(r['passed'] for r in resets),
        'planning_resets_checked':len(planning),
        'min_joint_margin_rad':min(min(r['geometry']['joint_margins_rad'].values()) for r in resets),
        'min_arm_table_clearance_m':min(min(v for n,v in r['geometry']['arm_table_m'].items() if n!='base_geom') for r in resets),
        'min_arm_block_clearance_m':min(min(r['geometry']['arm_block_m'].values()) for r in resets),
        'max_reset_attempts':max(r['attempts'] for r in resets),
        'dynamic_sequences':len(probes), 'dynamic_finite':all(r['finite'] for r in probes),
        'max_actuator_torque_error_Nm':max(r['max_actuator_torque_error_Nm'] for r in probes),
        'max_object_displacement_without_prior_contact_m':max(r['max_object_displacement_without_prior_contact_m'] for r in probes),
        'max_lock_position_drift_rad':max(r['max_lock_drift_rad'] for r in probes),
        'dynamic_min_arm_table_m':min(r['minimum_arm_table_m'] for r in probes),
        'dynamic_min_arm_block_m':min(r['minimum_arm_block_m'] for r in probes),
        'dynamic_min_joint_margin_rad':min(r['minimum_joint_margin_rad'] for r in probes),
        'zero_control_displacement_range_m':[min(r['object_displacement_m'] for r in probes if r['control_kind']=='zero'),max(r['object_displacement_m'] for r in probes if r['control_kind']=='zero')],
        'physical_performance_validated':False,
        'scale_run_authorized_by_this_report':False,
        'remaining':['Original XY-only fixed-height object does not establish gravity-loaded tabletop friction.',
            'Raw torque control and gravity-driven motion remain unchanged; development traces require interpretation.',
            'CPU/GPU parity at production batch size must pass before a new frozen scale protocol.',
            'Rebuild affected train/validation/test data and fit/select on the corrected distribution before performance claims.']}
    result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
