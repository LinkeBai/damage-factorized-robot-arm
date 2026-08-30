from pathlib import Path

import mujoco
import numpy as np
import torch

from robotarm.envs.constraint_lock import (
    activate_joint_lock,
    joint_lock_diagnostics,
    model_with_inactive_joint_locks,
)
from robotarm.models.topology_surgery import TopologySurgery


def test_analytic_state_projection_is_idempotent_for_random_batches():
    torch.manual_seed(11)
    surgery = TopologySurgery()
    state = torch.randn(32, 14); mask = torch.zeros(32, 5)
    mask[torch.arange(32), torch.arange(32) % 5] = 1.0
    angle = torch.randn(32, 5) * mask
    once = surgery.project_state(state, mask, angle)
    twice = surgery.project_state(once, mask, angle)
    torch.testing.assert_close(twice, once, rtol=0, atol=0)
    assert torch.count_nonzero(surgery.constraint_violation(twice, mask, angle)) == 0


ASSET = Path(__file__).resolve().parents[1] / "sim" / "assets" / "arm_push.xml"
MESH_ASSET = Path(__file__).resolve().parents[1] / "sim" / "assets" / "genkiarm_push.xml"
JOINTS = ("j1", "j2", "j3", "j4", "j5")


def test_solver_native_lock_produces_force_and_holds_joint() -> None:
    model = model_with_inactive_joint_locks(ASSET, JOINTS)
    data = mujoco.MjData(model)
    activate_joint_lock(model, data, "j3", 0.25)
    data.ctrl[model.actuator("m3").id] = 1.0
    for _ in range(100):
        mujoco.mj_step(model, data)
    diag = joint_lock_diagnostics(model, data, "j3")
    assert diag["position_violation"] < 2e-5
    assert diag["velocity_violation"] < 1e-4
    assert diag["locked_dof_constraint_force"] > 1e-5


def test_inactive_locks_do_not_constrain_other_joints() -> None:
    model = model_with_inactive_joint_locks(ASSET, JOINTS)
    data = mujoco.MjData(model)
    activate_joint_lock(model, data, "j3", 0.0)
    data.ctrl[model.actuator("m2").id] = 0.5
    for _ in range(100):
        mujoco.mj_step(model, data)
    j2 = model.joint("j2").id
    assert abs(float(data.qpos[int(model.jnt_qposadr[j2])])) > 1e-5
    assert np.isfinite(data.qpos).all()


def test_augmented_lock_model_resolves_relative_mesh_assets() -> None:
    model = model_with_inactive_joint_locks(MESH_ASSET, ("j2", "j3"))
    assert model.nmesh == 7
    assert model.equality("fault_lock_j2").id >= 0
