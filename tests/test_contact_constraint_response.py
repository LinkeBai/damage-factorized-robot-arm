import mujoco
import numpy as np

from scripts.diagnose_contact_constraint_response import generalized_force_by_type
from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks


def test_generalized_force_split_recovers_equality_component(tmp_path) -> None:
    path = tmp_path / "one.xml"
    path.write_text("""
    <mujoco><worldbody><body><joint name='j1'/><geom type='sphere' size='.1'/></body></worldbody>
    <actuator><motor name='m1' joint='j1'/></actuator></mujoco>
    """)
    model = model_with_inactive_joint_locks(path, ("j1",))
    data = mujoco.MjData(model)
    activate_joint_lock(model, data, "j1", 0.0)
    data.ctrl[model.actuator("m1").id] = 1.0
    mujoco.mj_step(model, data)
    equality = generalized_force_by_type(model, data, equality=True)
    other = generalized_force_by_type(model, data, equality=False)
    assert abs(float(equality[0])) > 1e-6
    assert np.isfinite(other).all()
