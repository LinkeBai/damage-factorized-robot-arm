"""Explicit supported-contact revision with substepped, stiffer constraints.

This is a simulation modeling choice, not measured hardware calibration.
Mass, shape, friction, damping, actuators, locks and initial geometry are retained.
Observation/action timing is set by the experiment at 5 ms, separately from the
internal numerical timestep. Original XML and both prior reset modules are intact.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import mujoco
import numpy as np
from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported
from robotarm.envs.constraint_lock import activate_joint_lock

JOINTS, PROFILES, XML = checked.JOINTS, checked.PROFILES, checked.XML
ResetSpec = checked.ResetSpec
VERSION = 'supported-resolved-contact-v2'
INTERNAL_DT = .00025
SOLREF = (.004, 1.)
SOLIMP = (.99, .9999, .001)


def make_model(lock, profile):
    model = supported.make_model(lock, profile)
    model.opt.timestep = INTERNAL_DT
    model.geom_solref[:] = SOLREF
    model.geom_solimp[:, :3] = SOLIMP
    model.jnt_solref[:] = SOLREF
    model.jnt_solimp[:, :3] = SOLIMP
    return model


@lru_cache(maxsize=4)
def equilibrium(profile, xml_sha):
    model = make_model(0, profile)
    data = mujoco.MjData(model)
    data.eq_active[:] = False
    mujoco.mj_step(model, data, nstep=round(1./INTERNAL_DT))
    mujoco.mj_forward(model, data)
    measurement = supported.support_measurement(model, data)
    geo = checked.geometry(model, data)
    if (measurement['normal_error_N'] > .005 or abs(measurement['z_velocity_m_s']) > 1e-5
            or min(geo['arm_block_m'].values()) <= 0.
            or min(v for n,v in geo['arm_table_m'].items() if n!='base_geom') <= 0.):
        raise RuntimeError('Fixed settling support failed')
    return {'duration_s': float(data.time), 'internal_steps': round(1./INTERNAL_DT),
            'settled_z_m': float(data.qpos[int(model.joint('block_z').qposadr[0])]),
            'support': measurement, 'xml_sha256': xml_sha}


def sample_reset(model, lock, profile, split, index, spec=ResetSpec(), seed=9132026):
    base = checked.make_model(lock, profile)
    source, record = checked.sample_reset(base, lock, profile, split, index, spec, seed)
    settled = equilibrium(profile, hashlib.sha256(XML.read_bytes()).hexdigest())
    data = mujoco.MjData(model)
    for name in JOINTS + ('block_x','block_y'):
        data.qpos[int(model.joint(name).qposadr[0])] = source.qpos[int(base.joint(name).qposadr[0])]
    data.qpos[int(model.joint('block_z').qposadr[0])] = settled['settled_z_m']
    data.qvel[:] = 0.
    activate_joint_lock(model, data, JOINTS[lock], record['lock_angle_rad'])
    audit = supported.inspect_reset(model, data, lock, spec)
    if not audit['passed']:
        raise RuntimeError('Resolved reset failed; no outcome-based retry: ' + str(audit['reasons']))
    return data, {**record, **audit, 'source_reset_id': record['reset_id'],
        'reset_id': VERSION + ':' + record['reset_id'], 'model_revision': VERSION,
        'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
        'joint_order': [model.joint(i).name for i in range(model.njnt)],
        'nq': model.nq, 'nv': model.nv, 'equilibrium': settled,
        'numerics': {'internal_timestep_s': INTERNAL_DT, 'geom_and_limit_solref': SOLREF,
                     'geom_and_limit_solimp_first_three': SOLIMP},
        'learning_projection': {'dimensions': 14, 'omitted': ['block_z','block_vz']}}
