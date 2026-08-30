from __future__ import annotations

import mujoco
import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv


def test_block_state_reports_linear_xy_velocity() -> None:
    env = MujocoArmEnv(
        xml_path="sim/assets/arm_push.xml", block_initial_xy=np.array([0.24, 0.10])
    )
    env.reset(target=np.array([0.28, 0.15, 0.02]))
    env.data.qvel[env._block_qvel_adr] = np.array([0.3, -0.2])
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(env.block_state()[2:], [0.3, -0.2])
