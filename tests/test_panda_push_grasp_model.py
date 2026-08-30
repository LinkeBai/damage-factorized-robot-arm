from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim" / "assets" / "panda_push_grasp.xml"


def test_official_panda_task_model_loads_with_arm_gripper_and_cube():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nu == 8
    for index in range(1, 8):
        assert model.joint(f"joint{index}").id >= 0
        assert model.actuator(f"actuator{index}").id >= 0
    for name in ("finger_joint1", "finger_joint2", "cube_free"):
        assert model.joint(name).id >= 0
    assert model.actuator("actuator8").id >= 0
    for name in ("eye_to_hand_left", "eye_to_hand_right"):
        assert model.camera(name).id >= 0


def test_panda_home_keyframe_and_gripper_are_dynamically_finite():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:] = model.key("home").ctrl
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert 0.0 <= data.joint("finger_joint1").qpos[0] <= 0.04
    assert 0.0 <= data.joint("finger_joint2").qpos[0] <= 0.04


def test_task_home_preserves_the_wrapper_cube_pose():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("task_home").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("task_cube").xpos, [0.50, 0.0, 0.025], atol=1e-12)


def test_both_eye_to_hand_cameras_point_toward_the_task_workspace():
    model = mujoco.MjModel.from_xml_path(str(MODEL)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    target = np.array([0.50, 0.0, 0.05])
    for name in ("eye_to_hand_left", "eye_to_hand_right"):
        camera_id = model.camera(name).id
        optical_axis = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        toward_target = target - data.cam_xpos[camera_id]
        cosine = float(np.dot(optical_axis, toward_target) / np.linalg.norm(toward_target))
        assert cosine > 0.75
