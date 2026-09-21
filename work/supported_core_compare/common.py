"""Shared identities and I/O for the isolated supported-contact comparison."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/lockpusher_supported_core_20260913'
PROTOCOL = OUT / 'protocol.json'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path, default=None):
    p = Path(path)
    if not p.exists() and default is not None:
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(p)


def load_protocol():
    return read(PROTOCOL)


def state14(model, data):
    """Read after mj_forward; caller owns synchronization and lock identity."""
    import numpy as np
    joints = ('j1', 'j2', 'j3', 'j4', 'j5')
    return np.concatenate((
        data.qpos[[int(model.joint(j).qposadr[0]) for j in joints]],
        data.qvel[[int(model.joint(j).dofadr[0]) for j in joints]],
        data.body('block').xpos[:2],
        data.qvel[[int(model.joint(j).dofadr[0]) for j in ('block_x', 'block_y')]],
    )).astype(np.float32)


def make_reset(lock, profile, split, index):
    from robotarm.envs import supported_push_reset as supported
    if load_protocol().get('numerics'):
        from robotarm.envs import resolved_push_reset as supported
    model = supported.make_model(lock, profile)
    data, metadata = supported.sample_reset(model, lock, profile, split, index, seed=9132026)
    return model, data, metadata


def step(model, data, on_substep=None):
    """Advance one fixed 5 ms observation interval; callback sees solver forces.

    Callback receives (model, data, zero_based_internal_index) immediately after
    each integration step, before this helper performs any forward calculation.
    The callback/caller owns post-integration mj_forward for derived positions.
    """
    import mujoco
    count = int(round(.005 / model.opt.timestep))
    if count < 1 or abs(count * model.opt.timestep - .005) > 1e-12:
        raise ValueError('Internal timestep must divide the fixed 5 ms observation interval')
    for index in range(count):
        mujoco.mj_step(model, data)
        if on_substep is not None:
            on_substep(model, data, index)
