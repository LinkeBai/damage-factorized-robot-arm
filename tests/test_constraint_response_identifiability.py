import mujoco
import numpy as np

from scripts.diagnose_constraint_response_identifiability import (
    equality_generalized_force,
    r2_score,
)
from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks


def test_r2_score_is_one_for_exact_prediction() -> None:
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]])
    assert r2_score(values, values) == 1.0


def test_equality_force_excludes_inactive_rows(tmp_path) -> None:
    xml = """
    <mujoco><worldbody><body><joint name='j1' type='hinge'/><geom type='sphere' size='.1'/></body></worldbody>
    <actuator><motor name='m1' joint='j1'/></actuator></mujoco>
    """
    path = tmp_path / "one.xml"
    path.write_text(xml)
    model = model_with_inactive_joint_locks(path, ("j1",))
    data = mujoco.MjData(model)
    activate_joint_lock(model, data, "j1", 0.0)
    data.ctrl[model.actuator("m1").id] = 1.0
    mujoco.mj_step(model, data)
    force = equality_generalized_force(model, data)
    assert force.shape == (model.nv,)
    assert abs(float(force[0])) > 1e-6
