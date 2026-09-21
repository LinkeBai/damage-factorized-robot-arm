"""Development initialization with gravity-loaded, fixed-orientation block support.

This is an explicit physics-model revision, not just a geometric reset fix.
The original asset, archived datasets, control algorithms and learning state stay
untouched. The optional 14-D learning projection hides block z and vz; callers
must audit those omitted motions before making any dynamics claim.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from robotarm.envs import checked_push_reset as checked
from robotarm.envs.constraint_lock import activate_joint_lock

JOINTS, PROFILES, XML = checked.JOINTS, checked.PROFILES, checked.XML
ResetSpec = checked.ResetSpec
VERSION = 'supported-contact-reset-development-v1'
SUPPORT_ATTRIBUTES = {'name': 'block_z', 'type': 'slide', 'axis': '0 0 1',
    'range': '-0.1 0.25', 'limited': 'true', 'armature': '0', 'damping': '0'}


def make_model(lock: int, profile: str):
    """Add only vertical object freedom plus the existing native lock definitions."""
    if lock not in range(5) or profile not in PROFILES:
        raise ValueError('Unknown lock or physics profile')
    root = ET.parse(XML).getroot()
    block = root.find("./worldbody/body[@name='block']")
    if block is None or block.find("joint[@name='block_z']") is not None:
        raise ValueError('Expected original block with x/y slides only')
    block.insert(2, ET.Element('joint', SUPPORT_ATTRIBUTES))
    compiler = root.find('compiler')
    if compiler is not None:
        for attribute in ('meshdir', 'texturedir', 'assetdir'):
            directory = compiler.get(attribute)
            if directory and not Path(directory).is_absolute():
                compiler.set(attribute, str((XML.parent / directory).resolve()))
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    # Same solver-native lock declarations as model_with_inactive_joint_locks.
    for name in JOINTS:
        ET.SubElement(equality, 'joint', {'name': 'fault_lock_' + name,
            'joint1': name, 'active': 'false', 'polycoef': '0 0 0 0 0',
            'solref': '0.002 1', 'solimp': '0.999 0.9999 0.001'})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    armv = [int(model.joint(n).dofadr[0]) for n in JOINTS]
    if profile in ('high_damping', 'mixed'):
        model.dof_damping[armv] *= 2.
    if profile in ('weak_motor', 'mixed'):
        model.actuator_gear[:, 0] *= .7
    return model


def support_measurement(model, data):
    normal, tangent = 0., 0.
    count = 0
    table, block = model.geom('table_geom').id, model.geom('block_geom').id
    for i, contact in enumerate(data.contact):
        if {int(contact.geom1), int(contact.geom2)} != {table, block}:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal += float(force[0])
        tangent += float(np.linalg.norm(force[1:3]))
        count += 1
    qz = int(model.joint('block_z').qposadr[0])
    vz = int(model.joint('block_z').dofadr[0])
    mg = float(model.body('block').mass[0] * -model.opt.gravity[2])
    return {'normal_force_N': normal, 'expected_mg_N': mg,
        'normal_error_N': abs(normal - mg), 'tangent_force_norm_sum_N': tangent,
        'table_contact_count': count, 'z_displacement_m': float(data.qpos[qz]),
        'z_velocity_m_s': float(data.qvel[vz]),
        'block_bottom_z_m': float(data.geom_xpos[block, 2] - model.geom_size[block, 2])}


@lru_cache(maxsize=16)
def _settled_support(profile: str, xml_sha256: str):
    """One fixed 1-second settling sequence per physics profile, never selection."""
    model = make_model(0, profile)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.eq_active[:] = False
    steps = int(round(1. / model.opt.timestep))
    if steps * model.opt.timestep > 1. + 1e-12:
        raise ValueError('Settling exceeds one second')
    rows = []
    for i in range(steps):
        data.ctrl[:] = 0.
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        g = checked.geometry(model, data)
        if min(g['arm_block_m'].values()) <= 0. or min(
                v for n, v in g['arm_table_m'].items() if n != 'base_geom') <= 0.:
            raise RuntimeError('Settling arm did not remain clear of block/table')
        rows.append({'time_s': float(data.time), **support_measurement(model, data)})
    final = rows[-1]
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or final['normal_error_N'] > .005 or abs(final['z_velocity_m_s']) > 1e-5
            or not -.00005 <= final['block_bottom_z_m'] <= 1e-8):
        raise RuntimeError(f'Fixed settling failed physical support checks: {final}')
    return {'xml_sha256': xml_sha256, 'profile': profile, 'steps': steps,
        'duration_s': float(data.time), 'initial_arm_q': [0.] * 5,
        'control': 'zero original motor inputs; no lock, PD or gravity compensation',
        'settled_z_m': final['z_displacement_m'],
        'max_transient_penetration_m': max(0., -min(r['block_bottom_z_m'] for r in rows)),
        'tail_normal_mean_N': float(np.mean([r['normal_force_N'] for r in rows[-40:]])),
        'rows': rows}


def inspect_reset(model, data, lock: int, spec: ResetSpec = ResetSpec()):
    # Reuse every original geometric/lock check, replacing only the original
    # exact-zero-height criterion with bounded steady soft support contact.
    audit = checked.inspect_reset(model, data, lock, spec)
    reasons = [r for r in audit['reasons'] if r != 'legacy_planar_object_height']
    support = support_measurement(model, data)
    if not -.00005 <= audit['geometry']['block_table_m'] <= 1e-8:
        reasons.append('support_surface_height')
    if support['normal_error_N'] > max(.005, .01 * support['expected_mg_N']):
        reasons.append('support_normal_load')
    if np.max(abs(data.qvel)) > 1e-12:
        reasons.append('nonzero_reset_velocity')
    jz = model.joint('block_z')
    if not jz.range[0] <= data.qpos[int(jz.qposadr[0])] <= jz.range[1]:
        reasons.append('vertical_joint_limit')
    return {'passed': not reasons, 'reasons': reasons,
            'geometry': audit['geometry'], 'support': support}


def sample_reset(model, lock: int, profile: str, split: str, index: int,
                 spec: ResetSpec = ResetSpec(), seed: int = 9132026):
    """Copy one checked geometry sample by joint name and install settled support.

    No trajectory is used to accept/reject or choose a reset. A failed final
    geometry/support audit raises immediately. This mutates model.eq_data:
    batched callers must save and restore each world's own lock angle.
    """
    base = checked.make_model(lock, profile)
    source, source_record = checked.sample_reset(base, lock, profile, split, index, spec, seed)
    equilibrium = _settled_support(profile, hashlib.sha256(XML.read_bytes()).hexdigest())
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for name in JOINTS + ('block_x', 'block_y'):
        data.qpos[int(model.joint(name).qposadr[0])] = source.qpos[int(base.joint(name).qposadr[0])]
    data.qpos[int(model.joint('block_z').qposadr[0])] = equilibrium['settled_z_m']
    data.qvel[:] = 0.
    activate_joint_lock(model, data, JOINTS[lock], source_record['lock_angle_rad'])
    audit = inspect_reset(model, data, lock, spec)
    if not audit['passed']:
        raise RuntimeError(f'Supported reset failed; no dynamic filtering/retry: {audit["reasons"]}')
    record = {**source_record, **audit, 'reset_id': VERSION + ':' + source_record['reset_id'],
        'source_reset_id': source_record['reset_id'], 'model_revision': VERSION,
        'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
        'joint_order': [model.joint(i).name for i in range(model.njnt)],
        'nq': model.nq, 'nv': model.nv,
        'equilibrium': {k: v for k, v in equilibrium.items() if k != 'rows'},
        'learning_projection': {'dimensions': 14, 'omitted': ['block_z', 'block_vz'],
            'global_dynamic_accuracy_validated': False}}
    return data, record


def settling_records():
    digest = hashlib.sha256(XML.read_bytes()).hexdigest()
    return [_settled_support(profile, digest) for profile in PROFILES]
