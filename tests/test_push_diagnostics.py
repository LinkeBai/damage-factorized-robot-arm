import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv


def test_named_contact_query_and_active_probe_shape():
    from scripts.run_push_benchmark import active_probe_action

    env = MujocoArmEnv(xml_path="sim/assets/arm_push.xml")
    env.reset(target=np.array([0.25, 0.15, 0.02]))
    assert isinstance(env.has_contact("tool_geom", "block_geom"), bool)
    action = active_probe_action(17, 2)
    assert action.shape == (5,)
    assert np.max(np.abs(action)) <= 0.7
