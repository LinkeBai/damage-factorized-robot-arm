from pathlib import Path

import mujoco
import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim" / "assets" / "genkiarm_push.xml"


def test_both_eye_to_hand_cameras_point_toward_the_task_workspace():
    model = mujoco.MjModel.from_xml_path(str(MODEL)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    target = np.array([0.22, 0.10, 0.05])
    for name in ("eye_to_hand_left", "eye_to_hand_right"):
        camera_id = model.camera(name).id
        optical_axis = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        toward_target = target - data.cam_xpos[camera_id]
        cosine = float(np.dot(optical_axis, toward_target) / np.linalg.norm(toward_target))
        assert cosine > 0.75


def test_genkiarm_push_model_loads_with_expected_task_contract():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 5
    for name in ("j1", "j2", "j3", "j4", "j5", "block_x", "block_y"):
        assert model.joint(name).id >= 0
    for name in ("eye_to_hand_left", "eye_to_hand_right"):
        assert model.camera(name).id >= 0


def test_genkiarm_push_uses_visual_cad_but_collision_proxies_for_contact():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    visual_ids = [model.geom(f"{name}_visual").id for name in
                  ("base", "link1", "link2", "link3", "link4", "link5", "tool")]
    assert np.all(model.geom_contype[visual_ids] == 0)
    assert np.all(model.geom_conaffinity[visual_ids] == 0)
    for name in ("tool_collision", "pusher_collision", "block_geom", "table_geom"):
        geom_id = model.geom(name).id
        assert model.geom_contype[geom_id] == 1
        assert model.geom_conaffinity[geom_id] == 1


def test_genkiarm_push_conforms_to_environment_state_contract():
    env = MujocoArmEnv(xml_path=MODEL, block_initial_xy=np.array([0.24, 0.10]))
    obs = env.reset(target=np.array([0.28, 0.12, 0.02]))
    assert obs["state"].shape == (14,)
    assert env.block_pos().shape == (2,)
    result = env.step(np.zeros(5))
    assert result["observation"]["state"].shape == (14,)
    assert np.isfinite(env.ee_pos()).all()


def test_genkiarm_push_has_reproducible_contact_and_block_motion():
    env = MujocoArmEnv(xml_path=MODEL, block_initial_xy=np.array([0.20, 0.10]))
    obs = env.reset(target=np.array([0.25, 0.10, 0.02]))
    initial = env.block_pos().copy()
    approach, _ = solve_reach_reference(
        np.array([0.17, 0.10, 0.025]), env.joint_ranges
    )
    push, _ = solve_reach_reference(
        np.array([0.28, 0.10, 0.020]), env.joint_ranges
    )
    contact_steps = 0
    for step in range(400):
        reference = approach if step < 160 else push
        action = joint_reference_action(obs["state"][:10], reference)
        result = env.step(action)
        obs = result["observation"]
        contact_steps += int(
            env.last_has_contact("tool_collision", "block_geom")
            or env.last_has_contact("pusher_collision", "block_geom")
        )
    assert contact_steps >= 20
    assert np.linalg.norm(env.block_pos() - initial) >= 0.02
