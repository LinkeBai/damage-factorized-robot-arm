"""Deterministic, geometry-checked resets for a new pushing protocol.

This module does not alter the archived XML, data, controllers or checkpoints.
Passing its reset checks is not a validation of contact dynamics or task gains.
Both training-data and planning callers must use the same constructor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np

from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks

ROOT = Path(__file__).resolve().parents[3]
XML = ROOT / 'sim/assets/arm_push.xml'
JOINTS = ('j1', 'j2', 'j3', 'j4', 'j5')
PROFILES = ('nominal', 'high_damping', 'weak_motor', 'mixed')
ARM_GEOMS = ('base_geom', 'shoulder_geom', 'upper_geom', 'forearm_geom',
             'wrist_pitch_geom', 'wrist_roll_geom', 'tool_geom', 'pusher_geom')
ASSEMBLY_PAIRS = {frozenset(p) for p in (
    ('shoulder_geom', 'upper_geom'), ('upper_geom', 'forearm_geom'),
    ('forearm_geom', 'wrist_pitch_geom'), ('wrist_pitch_geom', 'wrist_roll_geom'),
    ('tool_geom', 'pusher_geom'))}
SPLIT_CODES = {'development': 1, 'pool': 101, 'validation': 202, 'test': 303, 'planning': 404}


@dataclass(frozen=True)
class ResetSpec:
    version: str = 'checked-contact-reset-v1'
    # First geometry-feasible development candidate, not selected by task outcome.
    center_q: tuple[float, ...] = (0.42312146792525174, 0.6499092502434546,
        0.8384152721451945, 1.2142838836863876, 0.2291220269491827)
    q_std_rad: float = 0.015
    q_truncation_rad: float = 0.035
    joint_margin_rad: float = 0.05
    table_clearance_m: float = 0.006
    nonassembly_clearance_m: float = 0.003
    contact_gap_m: tuple[float, float] = (0.002, 0.004)
    workspace_halfwidth_m: float = 0.42
    max_attempts: int = 128


def make_model(lock: int, profile: str):
    if lock not in range(5) or profile not in PROFILES:
        raise ValueError('Unknown lock or physics profile')
    m = model_with_inactive_joint_locks(XML, JOINTS)
    armv = [int(m.joint(n).dofadr[0]) for n in JOINTS]
    if profile in ('high_damping', 'mixed'):
        m.dof_damping[armv] *= 2.0
    if profile in ('weak_motor', 'mixed'):
        m.actuator_gear[:, 0] *= 0.7
    # All equalities start inactive; the accepted reset supplies the lock angle.
    return m


def distance(m, d, a, b):
    return float(mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, 2., None))


def geometry(m, d):
    """Check visible robot geometry even when collision flags are disabled."""
    table = {n: distance(m, d, n, 'table_geom') for n in ARM_GEOMS}
    block = {n: distance(m, d, n, 'block_geom') for n in ARM_GEOMS}
    self_distances = {f'{a}/{b}': distance(m, d, a, b) for a, b in combinations(ARM_GEOMS, 2)}
    margins = {}
    for name in JOINTS:
        j = m.joint(name)
        q = float(d.qpos[int(j.qposadr[0])])
        margins[name] = float(min(q - j.range[0], j.range[1] - q))
    return {'arm_table_m': table, 'arm_block_m': block,
            'nonassembly_self_min_m': min(v for k, v in self_distances.items()
                if frozenset(k.split('/')) not in ASSEMBLY_PAIRS),
            'joint_margins_rad': margins,
            'block_table_m': distance(m, d, 'block_geom', 'table_geom')}


def inspect_reset(m, d, lock: int, spec: ResetSpec = ResetSpec()):
    g = geometry(m, d)
    reasons = []
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        reasons.append('nonfinite_state')
    if min(g['joint_margins_rad'].values()) < spec.joint_margin_rad:
        reasons.append('joint_margin')
    if min(v for n, v in g['arm_table_m'].items() if n != 'base_geom') < spec.table_clearance_m:
        reasons.append('arm_table_clearance')
    if g['arm_table_m']['base_geom'] < -1e-8:
        reasons.append('base_table_overlap')
    if min(g['arm_block_m'].values()) < spec.contact_gap_m[0] - 1e-8:
        reasons.append('arm_block_clearance')
    if g['arm_block_m']['pusher_geom'] > spec.contact_gap_m[1] + 0.002:
        reasons.append('pusher_too_far_from_object')
    if g['nonassembly_self_min_m'] < spec.nonassembly_clearance_m:
        reasons.append('nonassembly_self_clearance')
    if abs(g['block_table_m']) > 1e-8:
        reasons.append('legacy_planar_object_height')
    if max(abs(d.body('block').xpos[:2])) > spec.workspace_halfwidth_m:
        reasons.append('object_outside_workspace')
    for n in ('block_x', 'block_y'):
        j = m.joint(n)
        if not j.range[0] <= d.qpos[int(j.qposadr[0])] <= j.range[1]:
            reasons.append('object_joint_limit')
    j = m.joint(JOINTS[lock]); eq = m.equality('fault_lock_' + JOINTS[lock]).id
    if not d.eq_active[eq] or np.count_nonzero(d.eq_active) != 1:
        reasons.append('wrong_active_lock')
    if abs(d.qpos[int(j.qposadr[0])] - m.eq_data[eq, 0]) > 1e-10:
        reasons.append('lock_angle_mismatch')
    if abs(d.qvel[int(j.dofadr[0])]) > 1e-10:
        reasons.append('nonzero_locked_velocity')
    if d.time != 0:
        reasons.append('nonzero_reset_time')
    return {'passed': not reasons, 'reasons': reasons, 'geometry': g}


def sample_reset(m, lock: int, profile: str, split: str, index: int,
                 spec: ResetSpec = ResetSpec(), seed: int = 9132026):
    """Accept by geometry alone; never consult a learned model or task result.

    Each identity has its own deterministic rejection stream, independent of
    batch size and ordering. Attempt counts remain visible in the returned record.
    Initial velocities are zero. Raw motors, gravity and the original planar
    object dynamics are unchanged.
    """
    if split not in SPLIT_CODES or lock not in range(5) or profile not in PROFILES or index < 0:
        raise ValueError('Invalid reset identity')
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES[split], lock,
        PROFILES.index(profile), index]))
    d = mujoco.MjData(m)
    armq = np.array([int(m.joint(n).qposadr[0]) for n in JOINTS])
    blockq = np.array([int(m.joint(n).qposadr[0]) for n in ('block_x', 'block_y')])
    origin = m.body('block').pos[:2].copy()

    def set_pose(q, xy):
        mujoco.mj_resetData(m, d)
        d.qpos[armq] = q
        d.qpos[blockq] = np.asarray(xy) - origin
        d.qvel[:] = 0
        activate_joint_lock(m, d, JOINTS[lock], float(q[lock]))

    rejection_counts = {}
    for attempt in range(1, spec.max_attempts + 1):
        noise = rng.normal(0., spec.q_std_rad, 5)
        if np.any(abs(noise) > spec.q_truncation_rad):
            rejection_counts['q_truncation'] = rejection_counts.get('q_truncation', 0) + 1
            continue
        q = np.asarray(spec.center_q) + noise
        set_pose(q, [.4, .4])
        g = geometry(m, d)
        if (min(g['joint_margins_rad'].values()) < spec.joint_margin_rad or
            min(v for n, v in g['arm_table_m'].items() if n != 'base_geom') < spec.table_clearance_m or
            g['nonassembly_self_min_m'] < spec.nonassembly_clearance_m):
            rejection_counts['arm_geometry'] = rejection_counts.get('arm_geometry', 0) + 1
            continue
        gid = m.geom('pusher_geom').id
        axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
        p0 = d.geom_xpos[gid] - m.geom_size[gid, 1] * axis
        p1 = d.geom_xpos[gid] + m.geom_size[gid, 1] * axis
        dz = p1[2] - p0[2]
        if min(p0[2], p1[2]) > .036 or max(p0[2], p1[2]) < .012:
            rejection_counts['pusher_height'] = rejection_counts.get('pusher_height', 0) + 1
            continue
        t = float(np.clip((.024 - p0[2]) / dz, 0, 1)) if abs(dz) > 1e-10 else .5
        anchor = p0 + t * (p1 - p0)
        angle = np.arctan2(anchor[1], anchor[0]) + rng.uniform(-.65, .65)
        direction = np.array([np.cos(angle), np.sin(angle)])
        target_gap = float(rng.uniform(*spec.contact_gap_m))
        lo, hi = 0., .12
        set_pose(q, anchor[:2])
        if min(distance(m, d, n, 'block_geom') for n in ('tool_geom', 'pusher_geom')) > target_gap:
            rejection_counts['missing_inner_bracket'] = rejection_counts.get('missing_inner_bracket', 0) + 1
            continue
        set_pose(q, anchor[:2] + hi * direction)
        if min(distance(m, d, n, 'block_geom') for n in ('tool_geom', 'pusher_geom')) < target_gap:
            rejection_counts['missing_outer_bracket'] = rejection_counts.get('missing_outer_bracket', 0) + 1
            continue
        for _ in range(20):
            r = .5 * (lo + hi)
            set_pose(q, anchor[:2] + r * direction)
            if min(distance(m, d, n, 'block_geom') for n in ('tool_geom', 'pusher_geom')) < target_gap:
                lo = r
            else:
                hi = r
        set_pose(q, anchor[:2] + hi * direction)
        audit = inspect_reset(m, d, lock, spec)
        if audit['passed']:
            return d, {'reset_id': f'{spec.version}-{split}-D{lock+1}-P{PROFILES.index(profile)}-{index:06d}',
                       'seed': seed, 'attempts': attempt, 'rejections': rejection_counts,
                       'target_gap_m': target_gap, 'lock_angle_rad': float(q[lock]),
                       'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist(),
                       'spec': asdict(spec), **audit}
        for reason in audit['reasons']:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    raise RuntimeError(f'No valid reset after {spec.max_attempts} attempts: {split}/{lock}/{profile}/{index}; {rejection_counts}')


def planning_reset(lock: int, profile: str, band: int, index: int,
                   spec: ResetSpec = ResetSpec(), seed: int = 9132026):
    if band not in (0, 1):
        raise ValueError('Unknown goal-distance band')
    m = make_model(lock, profile)
    d, record = sample_reset(m, lock, profile, 'planning', 2 * index + band, spec, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 505, lock, PROFILES.index(profile), band, index]))
    radius = float(rng.uniform(*((.04, .065) if band == 0 else (.065, .09))))
    angle = float(rng.uniform(-np.pi / 4, np.pi / 4))
    goal = d.body('block').xpos[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
    if max(abs(goal)) + .02 >= .5:
        raise RuntimeError('Goal leaves the table; change the development protocol, not a test outcome')
    return m, d, goal, record
