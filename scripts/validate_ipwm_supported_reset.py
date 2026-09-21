"""Bounded development initialization validation, never model-performance results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mujoco
import numpy as np
from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported
from scripts import validate_ipwm_checked_reset as existing


def dynamic_probe(model, initial, lock, kind):
    data = existing.clone(model, initial)
    start = existing.state(model, data)
    control = np.zeros(5)
    if kind != 'zero':
        control[next(i for i in range(5) if i != lock)] = .2 if kind == 'positive' else -.2
    rows = []
    ever_object_contact = False
    no_contact_displacement = 0.
    for step in range(51):
        data.ctrl[:] = control
        mujoco.mj_forward(model, data)
        geo = checked.geometry(model, data)
        support = supported.support_measurement(model, data)
        contacts = []
        for ci, contact in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, force)
            names = [model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name]
            contacts.append({'geoms': names, 'distance_m': float(contact.dist),
                             'normal_force_N': float(force[0])})
            if ('block_geom' in names and set(names) & {'tool_geom', 'pusher_geom'}
                    and force[0] > 1e-9):
                ever_object_contact = True
        state = existing.state(model, data)
        displacement = float(np.linalg.norm(state[10:12] - start[10:12]))
        if not ever_object_contact:
            no_contact_displacement = max(no_contact_displacement, displacement)
        rows.append({'step': step, 'time_s': float(data.time), 'state14': state.tolist(),
            'ctrl': control.tolist(), 'support': support, 'contacts': contacts,
            'finite': bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            'joint_margins_rad': geo['joint_margins_rad'],
            'min_arm_table_m': min(v for n, v in geo['arm_table_m'].items() if n != 'base_geom'),
            'min_arm_block_m': min(geo['arm_block_m'].values()),
            'nonassembly_self_min_m': geo['nonassembly_self_min_m'],
            'lock_drift_rad': float(abs(state[lock] - start[lock])),
            'lock_speed_rad_s': float(abs(state[5 + lock])),
            'object_displacement_m': displacement})
        if step < 50:
            mujoco.mj_step(model, data)
    return {'control_kind': kind, 'steps': 50, 'rows': rows,
        'finite': all(r['finite'] for r in rows),
        'object_displacement_m': rows[-1]['object_displacement_m'],
        'max_object_displacement_without_prior_contact_m': no_contact_displacement,
        'z_displacement_range_m': [min(r['support']['z_displacement_m'] for r in rows),
                                 max(r['support']['z_displacement_m'] for r in rows)],
        'max_abs_z_velocity_m_s': max(abs(r['support']['z_velocity_m_s']) for r in rows),
        'max_lock_drift_rad': max(r['lock_drift_rad'] for r in rows),
        'min_arm_table_m': min(r['min_arm_table_m'] for r in rows),
        'min_arm_block_m': min(r['min_arm_block_m'] for r in rows),
        'min_joint_margin_rad': min(min(r['joint_margins_rad'].values()) for r in rows)}


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (checked.XML, ROOT/'src/robotarm/envs/checked_push_reset.py',
            ROOT/'src/robotarm/envs/supported_push_reset.py',
            ROOT/'src/robotarm/envs/constraint_lock.py', Path(__file__),
            ROOT/'scripts/validate_ipwm_checked_reset.py')}


def gpu_validation(output):
    import importlib.metadata
    # Explicit, scoped callback replacement reuses the already reviewed per-world
    # eq_data batching and restores each CPU world's lock before every step.
    old_make, old_sample = existing.make_model, existing.sample_reset
    existing.make_model, existing.sample_reset = supported.make_model, supported.sample_reset
    rows = []
    try:
        for lock in range(5):
            for profile in checked.PROFILES:
                row = existing.gpu_parity(lock, profile, count=4)
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        existing.make_model, existing.sample_reset = old_make, old_sample
    report = {'scope': 'development initialization validation: 80 paired CPU/GPU trajectories, 4 worlds per cell',
        'source_sha256': source_hashes(), 'rows': rows, 'passed': all(r['passed'] for r in rows),
        'mujoco_version': mujoco.__version__,
        'mujoco_warp_version': importlib.metadata.version('mujoco-warp'),
        'warp_version': importlib.metadata.version('warp-lang'),
        'paired_worlds': 80, 'steps_per_world': 50,
        'projection': '14-D arm/planar object parity; extra z/vz audited in CPU dynamic-probes.json',
        'production_batch_validated': False, 'model_performance_validated': False,
        'global_dynamic_accuracy_validated': False}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Development CPU/GPU parity failed; preserved all results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'runs/ipwm_supported_reset_development')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['gpu-parity.json'] if args.gpu else ['validation.json', 'reset-records.json', 'dynamic-probes.json', 'settling-records.json']
    for name in names:
        if (args.output/name).exists():
            raise FileExistsError(f'Preserve prior development evidence: {args.output/name}')
    if args.gpu:
        gpu_validation(args.output/'gpu-parity.json')
        return
    before = source_hashes()
    resets, probes = [], []
    for lock in range(5):
        for profile in checked.PROFILES:
            model = supported.make_model(lock, profile)
            for index in range(4):
                data, record = supported.sample_reset(model, lock, profile, 'development', index)
                if not supported.inspect_reset(model, data, lock)['passed']:
                    raise RuntimeError('Post-copy reset audit failed')
                resets.append({'lock': lock + 1, 'profile': profile, **record})
                if index == 0:
                    for kind in ('zero', 'positive', 'negative'):
                        probes.append({'lock': lock + 1, 'profile': profile,
                            'reset_id': record['reset_id'], **dynamic_probe(model, data, lock, kind)})
            print(f'checked supported initialization D{lock+1} {profile}', flush=True)
    assert before == source_hashes()
    report = {'scope': 'development initialization validation; not model performance results',
        'model_revision': supported.VERSION, 'source_sha256': before,
        'mujoco_version': mujoco.__version__, 'geometric_resets_checked': len(resets),
        'geometry_and_support_passed': all(r['passed'] for r in resets),
        'nq': 8, 'nv': 8, 'fixed_block_orientation': True,
        'only_physics_change': {'added_block_slide': supported.SUPPORT_ATTRIBUTES},
        'reset_velocity': 'all zero after settled support height is copied; lock activated at sampled angle',
        'min_joint_margin_rad': min(min(r['geometry']['joint_margins_rad'].values()) for r in resets),
        'min_arm_table_clearance_m': min(min(v for n, v in r['geometry']['arm_table_m'].items() if n != 'base_geom') for r in resets),
        'min_arm_block_clearance_m': min(min(r['geometry']['arm_block_m'].values()) for r in resets),
        'reset_block_table_range_m': [min(r['geometry']['block_table_m'] for r in resets), max(r['geometry']['block_table_m'] for r in resets)],
        'reset_normal_load_range_N': [min(r['support']['normal_force_N'] for r in resets), max(r['support']['normal_force_N'] for r in resets)],
        'max_reset_normal_load_error_N': max(r['support']['normal_error_N'] for r in resets),
        'settling_steps': sum(r['steps'] for r in supported.settling_records()),
        'dynamic_sequences': len(probes), 'dynamic_mj_steps': len(probes) * 50,
        'dynamic_finite': all(p['finite'] for p in probes),
        'dynamic_z_displacement_range_m': [min(p['z_displacement_range_m'][0] for p in probes), max(p['z_displacement_range_m'][1] for p in probes)],
        'dynamic_max_abs_z_velocity_m_s': max(p['max_abs_z_velocity_m_s'] for p in probes),
        'dynamic_min_joint_margin_rad': min(p['min_joint_margin_rad'] for p in probes),
        'dynamic_min_arm_table_m': min(p['min_arm_table_m'] for p in probes),
        'dynamic_min_arm_block_m': min(p['min_arm_block_m'] for p in probes),
        'dynamic_max_lock_drift_rad': max(p['max_lock_drift_rad'] for p in probes),
        'max_object_displacement_without_prior_contact_m': max(p['max_object_displacement_without_prior_contact_m'] for p in probes),
        'zero_control_object_displacement_range_m': [min(p['object_displacement_m'] for p in probes if p['control_kind'] == 'zero'), max(p['object_displacement_m'] for p in probes if p['control_kind'] == 'zero')],
        'model_performance_validated': False, 'global_dynamic_accuracy_validated': False,
        'production_batch_validated': False, 'scale_run_authorized_by_this_report': False,
        'limitations': ['14-D learning projection omits z and vz; observed vertical motion is recorded, not assumed away.',
            'The block orientation remains fixed, so tipping and rotational contact behavior are excluded.',
            'Original raw motors, gravity, damping and friction remain; gravity-driven robot motion is not repaired by object support.',
            'No dynamic outcome was used to select, reject or retry a reset.',
            '80 worlds at batch size four do not validate production batches or learned-model performance.']}
    for name, value in [('reset-records.json', resets), ('dynamic-probes.json', probes),
                        ('settling-records.json', supported.settling_records()), ('validation.json', report)]:
        (args.output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
