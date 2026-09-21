"""Development-only contact/limit resolution check; no learned-model scoring."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, PROTOCOL, read, sha, write
import mujoco
import numpy as np
from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported


def configure(model, kind):
    model.opt.timestep = .001
    if kind == 'stiff_contact_limit_1ms':
        model.geom_solref[:] = [.004, 1.]
        model.geom_solimp[:, :3] = [.99, .9999, .001]
        model.jnt_solref[:] = [.004, 1.]
        model.jnt_solimp[:, :3] = [.99, .9999, .001]


@lru_cache(maxsize=8)
def settled_z(kind, profile):
    model = supported.make_model(0, profile)
    configure(model, kind)
    data = mujoco.MjData(model)
    data.eq_active[:] = False
    mujoco.mj_step(model, data, nstep=1000)
    mujoco.mj_forward(model, data)
    measurement = supported.support_measurement(model, data)
    if measurement['normal_error_N'] > .005 or abs(measurement['z_velocity_m_s']) > 1e-5:
        raise RuntimeError('Candidate support equilibrium failed')
    return float(data.qpos[int(model.joint('block_z').qposadr[0])])


def run(kind, record, source_q, source_v, actions):
    lock, profile = record['lock'], record['profile']
    model = supported.make_model(lock, profile)
    configure(model, kind)
    data = mujoco.MjData(model)
    data.qpos[:] = source_q
    data.qvel[:] = source_v
    data.qpos[int(model.joint('block_z').qposadr[0])] = settled_z(kind, profile)
    from robotarm.envs.constraint_lock import activate_joint_lock
    activate_joint_lock(model, data, checked.JOINTS[lock], record['reset_record']['lock_angle_rad'])
    audit = supported.inspect_reset(model, data, lock)
    if not audit['passed']:
        raise RuntimeError('Candidate initial geometry/support failed')
    rows = []
    for step in range(251):
        if step:
            data.ctrl[:] = actions[(step-1)//50]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        geo = checked.geometry(model, data)
        support = supported.support_measurement(model, data)
        rows.append([float(data.time), min(geo['joint_margins_rad'].values()),
                     min(v for n,v in geo['arm_table_m'].items() if n != 'base_geom'),
                     min(geo['arm_block_m'].values()), geo['block_table_m'],
                     support['z_displacement_m'], support['z_velocity_m_s'],
                     *data.qpos, *data.qvel])
    values = np.asarray(rows)
    return values, {'identity': record['reset_record']['reset_id'], 'control_kind': record['control_kind'],
        'initial_passed': True, 'finite': bool(np.isfinite(values).all()),
        'min_joint_margin_rad': float(values[:,1].min()),
        'min_arm_table_m': float(values[:,2].min()),
        'min_arm_block_m': float(values[:,3].min()),
        'min_block_table_m': float(values[:,4].min()),
        'z_min_m': float(values[:,5].min()), 'z_max_m': float(values[:,5].max()),
        'max_abs_vz_m_s': float(abs(values[:,6]).max()),
        'warnings': {str(i): int(w.number) for i,w in enumerate(data.warning) if w.number}}


def main():
    folder = OUT/'solver-development'
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    source = OUT/'data-development'
    records = read(source/'records.json')
    spec = {'scope': 'Physics/numerics development only, all original 320 identities and actions.',
            'protocol_sha256': sha(PROTOCOL), 'source_data_sha256': sha(source/'data.npz'),
            'source_records_sha256': sha(source/'records.json'), 'script_sha256': sha(__file__),
            'candidates': ['unchanged_softness_1ms', 'stiff_contact_limit_1ms'],
            'observation_dt_s': .005, 'internal_dt_s': .001,
            'fixed_trajectory_duration_s': .25,
            'stiff_candidate': {'geom_and_joint_limit_solref': [.004, 1.],
                                'solimp_first_three': [.99, .9999, .001]},
            'unchanged': ['mass', 'geometry', 'friction', 'damping', 'actuators', 'locks', 'actions', 'initial joint pose'],
            'selection_basis': 'Numerical/contact/limit behavior only; no model or task-performance comparison.'}
    write(folder/'protocol.json', spec)
    with np.load(source/'data.npz', allow_pickle=False) as a:
        q, v, actions = a['full_qpos'][:,0], a['full_qvel'][:,0], a['segment_actions']
    for kind in spec['candidates']:
        traces, rows = [], []
        for i, record in enumerate(records):
            trace, row = run(kind, record, q[i], v[i], actions[i])
            traces.append(trace); rows.append(row)
        np.savez_compressed(folder/(kind+'.npz'), traces=np.asarray(traces))
        report = {'candidate': kind, 'rows': rows, 'trace_sha256': sha(folder/(kind+'.npz')),
                  'summary': {key: min(r[key] for r in rows) for key in
                              ['min_joint_margin_rad','min_arm_table_m','min_arm_block_m','min_block_table_m','z_min_m']}}
        report['summary'].update(z_max_m=max(r['z_max_m'] for r in rows),
                                  max_abs_vz_m_s=max(r['max_abs_vz_m_s'] for r in rows),
                                  all_finite=all(r['finite'] for r in rows))
        write(folder/(kind+'.json'), report)
        print(json.dumps({'candidate': kind, **report['summary']}), flush=True)


if __name__ == '__main__':
    main()
