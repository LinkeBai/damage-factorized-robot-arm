from pathlib import Path

import mujoco
import numpy as np
import torch

from robotarm.envs.damage import DamageConfig
from robotarm.training.variable_trajectory import (
    collate_variable_trajectories,
    make_single_transition,
    observe_mujoco_nodes,
)


ROOT = Path(__file__).resolve().parents[1]


def _one_step(path: Path, joints: tuple[str, ...], object_body: str, robot: str):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = observe_mujoco_nodes(model, data, joint_names=joints, object_body=object_body)
    mujoco.mj_step(model, data)
    after = observe_mujoco_nodes(model, data, joint_names=joints, object_body=object_body)
    return make_single_transition(
        robot=robot, task="push", joint_names=joints, before=before, after=after,
        action=np.zeros(len(joints)), applied_action=np.zeros(len(joints)),
        damage=DamageConfig.intact(len(joints)), contact=False,
    )


def test_real_genkiarm_and_panda_states_share_one_lossless_contract():
    genki = _one_step(
        ROOT / "sim/assets/genkiarm_push.xml", tuple(f"j{i}" for i in range(1, 6)),
        "block", "genkiarm",
    )
    panda = _one_step(
        ROOT / "sim/assets/panda_push_grasp.xml", tuple(f"joint{i}" for i in range(1, 8)),
        "task_cube", "panda",
    )
    assert genki.dof == 5 and panda.dof == 7
    assert genki.object_pose.shape[-1] == panda.object_pose.shape[-1] == 7
    assert genki.object_twist.shape[-1] == panda.object_twist.shape[-1] == 6
    batch = collate_variable_trajectories([genki, panda])
    assert batch["joint_state"].shape == (2, 2, 7, 2)
    torch.testing.assert_close(batch["joint_state"][0, :, :5], genki.joint_state)
    torch.testing.assert_close(batch["joint_state"][1, :, :7], panda.joint_state)
    assert batch["node_valid"].tolist() == [
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True],
    ]


def test_contract_rejects_non_projected_locked_transition():
    trajectory = _one_step(
        ROOT / "sim/assets/genkiarm_push.xml", tuple(f"j{i}" for i in range(1, 6)),
        "block", "genkiarm",
    )
    trajectory.lock_mask[2] = 1
    trajectory.lock_angle[2] = 0.5
    try:
        trajectory.validate()
    except ValueError as error:
        assert "locked coordinate" in str(error)
    else:
        raise AssertionError("invalid locked transition was accepted")
