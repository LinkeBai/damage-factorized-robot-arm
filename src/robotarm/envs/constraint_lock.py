"""Solver-native joint locks for constraint-response mechanism diagnostics.

The deployed environment keeps analytic post-step projection for exact safety.
This module instead creates solver-visible equality constraints so that lock
reaction forces can be measured during training-only feasibility studies.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


def model_with_inactive_joint_locks(
    xml_path: str | Path,
    joint_names: tuple[str, ...],
) -> mujoco.MjModel:
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()
    # ``from_xml_string`` has no source-file directory for resolving assets.
    # Preserve MJCF path semantics by making relative compiler directories
    # absolute before serializing the augmented model.
    compiler = root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir"):
            directory = compiler.attrib.get(attribute)
            if directory and not Path(directory).is_absolute():
                compiler.attrib[attribute] = str((xml_path.parent / directory).resolve())
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for joint_name in joint_names:
        ET.SubElement(equality, "joint", {
            "name": f"fault_lock_{joint_name}",
            "joint1": joint_name,
            "active": "false",
            "polycoef": "0 0 0 0 0",
            "solref": "0.002 1",
            "solimp": "0.999 0.9999 0.001",
        })
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def activate_joint_lock(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    lock_angle: float,
) -> int:
    equality_id = model.equality(f"fault_lock_{joint_name}").id
    model.eq_data[equality_id, :] = 0.0
    model.eq_data[equality_id, 0] = float(lock_angle)
    if hasattr(data, "eq_active"):
        data.eq_active[:] = 0
        data.eq_active[equality_id] = 1
    else:
        model.eq_active0[:] = 0
        model.eq_active0[equality_id] = 1
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return equality_id


def joint_lock_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> dict[str, float]:
    joint_id = model.joint(joint_name).id
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    equality_id = model.equality(f"fault_lock_{joint_name}").id
    lock_angle = float(model.eq_data[equality_id, 0])
    return {
        "position_violation": abs(float(data.qpos[qpos_address]) - lock_angle),
        "velocity_violation": abs(float(data.qvel[dof_address])),
        "locked_dof_constraint_force": abs(float(data.qfrc_constraint[dof_address])),
        "constraint_force_norm": float(np.linalg.norm(data.qfrc_constraint)),
    }
