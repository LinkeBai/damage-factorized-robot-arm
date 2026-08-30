from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.collect_warp import _named_state_addresses


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("asset", ["arm_push.xml", "genkiarm_push.xml"])
def test_named_state_addresses_preserve_world_model_contract(asset: str):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "assets" / asset))
    arm_qpos, arm_qvel, block_qpos, block_qvel = _named_state_addresses(model)

    assert arm_qpos.shape == arm_qvel.shape == (5,)
    assert block_qpos.shape == block_qvel.shape == (2,)
    assert len(np.unique(np.concatenate((arm_qpos, block_qpos)))) == 7
    assert len(np.unique(np.concatenate((arm_qvel, block_qvel)))) == 7
    for index, name in enumerate(("j1", "j2", "j3", "j4", "j5")):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert arm_qpos[index] == model.jnt_qposadr[joint_id]
        assert arm_qvel[index] == model.jnt_dofadr[joint_id]


def test_named_state_addresses_reject_missing_contract_joint(tmp_path: Path):
    xml = tmp_path / "incomplete.xml"
    xml.write_text(
        '<mujoco><worldbody><body><joint name="j1" type="hinge"/>'
        '<geom type="sphere" size="0.01" mass="0.1"/></body>'
        '</worldbody></mujoco>', encoding="utf-8"
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    with pytest.raises(ValueError, match="j2"):
        _named_state_addresses(model)
