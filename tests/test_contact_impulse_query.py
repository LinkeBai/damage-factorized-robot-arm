import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from scripts.run_push_benchmark import PUSH_XML


def test_named_contact_impulse_has_planar_shape() -> None:
    env = MujocoArmEnv(xml_path=PUSH_XML)
    env.reset(target=np.array([0.25, 0.10, 0.02]))
    impulse = env.contact_impulse_xy("pusher_geom", "block_geom")
    records = env.contact_records("pusher_geom", "block_geom")
    assert impulse.shape == (2,)
    assert np.all(np.isfinite(impulse))
    assert isinstance(records, list)
