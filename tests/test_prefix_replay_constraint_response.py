from pathlib import Path

import mujoco
import numpy as np

from scripts.diagnose_prefix_replay_constraint_response import _copy_data


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim" / "assets" / "genkiarm_push.xml"


def test_copy_data_preserves_full_physical_prefix():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    source = mujoco.MjData(model)
    source.qpos[:] = np.linspace(-0.1, 0.1, model.nq)
    source.qvel[:] = np.linspace(-0.2, 0.2, model.nv)
    source.ctrl[:] = np.linspace(-0.3, 0.3, model.nu)
    mujoco.mj_forward(model, source)
    copied = _copy_data(model, source)
    np.testing.assert_array_equal(copied.qpos, source.qpos)
    np.testing.assert_array_equal(copied.qvel, source.qvel)
    np.testing.assert_array_equal(copied.ctrl, source.ctrl)
    np.testing.assert_array_equal(copied.act, source.act)
