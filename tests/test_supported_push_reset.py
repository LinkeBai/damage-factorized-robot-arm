import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import pytest

from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported


@pytest.mark.parametrize('lock,profile', [(0, 'nominal'), (2, 'mixed'), (4, 'weak_motor')])
def test_supported_model_preserves_original_physics_and_named_state(lock, profile):
    original_bytes = checked.XML.read_bytes()
    base = checked.make_model(lock, profile)
    original, _ = checked.sample_reset(base, lock, profile, 'development', 0)
    model = supported.make_model(lock, profile)
    data, record = supported.sample_reset(model, lock, profile, 'development', 0)
    assert model.nq == base.nq + 1 and model.nv == base.nv + 1
    for name in checked.JOINTS + ('block_x', 'block_y'):
        a, b = base.joint(name), model.joint(name)
        assert data.qpos[int(b.qposadr[0])] == original.qpos[int(a.qposadr[0])]
        np.testing.assert_array_equal(a.range, b.range)
        np.testing.assert_array_equal(base.dof_damping[a.dofadr], model.dof_damping[b.dofadr])
        np.testing.assert_array_equal(base.dof_armature[a.dofadr], model.dof_armature[b.dofadr])
    np.testing.assert_array_equal(base.actuator_gear, model.actuator_gear)
    np.testing.assert_array_equal(base.body_mass, model.body_mass)
    np.testing.assert_array_equal(base.geom_friction, model.geom_friction)
    assert model.opt.timestep == base.opt.timestep
    assert checked.XML.read_bytes() == original_bytes
    assert supported.inspect_reset(model, data, lock)['passed']
    assert record['support']['normal_error_N'] < .005
    assert record['learning_projection']['omitted'] == ['block_z', 'block_vz']


def test_supported_reset_rejects_unloaded_floating_object():
    model = supported.make_model(2, 'nominal')
    data, _ = supported.sample_reset(model, 2, 'nominal', 'development', 0)
    data.qpos[int(model.joint('block_z').qposadr[0])] = .005
    mujoco.mj_forward(model, data)
    audit = supported.inspect_reset(model, data, 2)
    assert not audit['passed']
    assert {'support_surface_height', 'support_normal_load'} <= set(audit['reasons'])
