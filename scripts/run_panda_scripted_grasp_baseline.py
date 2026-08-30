"""Physical-contact Panda Grasp feasibility baseline (no weld, no learned policy)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]


def _arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos, dof = [], []
    for i in range(1, 8):
        joint = model.joint(f"joint{i}")
        qpos.append(int(joint.qposadr[0])); dof.append(int(joint.dofadr[0]))
    return np.asarray(qpos), np.asarray(dof)


def _finger_cube_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    cube_body = model.body("task_cube").id
    left_body, right_body = model.body("left_finger").id, model.body("right_finger").id
    left = right = False
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        left |= bodies == {cube_body, left_body}
        right |= bodies == {cube_body, right_body}
    return left, right


def _position_step(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray,
                   desired_quat: np.ndarray, arm_qpos: np.ndarray,
                   arm_dof: np.ndarray, gain: float = 0.35) -> None:
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, model.body("hand").id)
    jac = np.concatenate((jacp[:, arm_dof], jacr[:, arm_dof]), axis=0)
    current_quat = np.empty(4); rotation_error = np.empty(3)
    mujoco.mju_mat2Quat(current_quat, data.body("hand").xmat)
    mujoco.mju_subQuat(rotation_error, desired_quat, current_quat)
    error = np.concatenate((target - data.body("hand").xpos, 0.6 * rotation_error))
    damping = 2e-3
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
    dq = np.clip(dq, -0.08, 0.08)
    desired = data.qpos[arm_qpos] + gain * dq
    data.ctrl[:7] = np.clip(desired, model.actuator_ctrlrange[:7, 0], model.actuator_ctrlrange[:7, 1])


def _solve_pose_ik(model: mujoco.MjModel, q_reference: np.ndarray, target: np.ndarray,
                   desired_quat: np.ndarray, arm_qpos: np.ndarray) -> np.ndarray:
    scratch = mujoco.MjData(model)
    lower = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 0]
    upper = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 1]

    def residual(q: np.ndarray) -> np.ndarray:
        scratch.qpos[:] = model.key("task_home").qpos
        scratch.qpos[arm_qpos] = q
        mujoco.mj_forward(model, scratch)
        current = np.empty(4); rotation = np.empty(3)
        mujoco.mju_mat2Quat(current, scratch.body("hand").xmat)
        mujoco.mju_subQuat(rotation, desired_quat, current)
        regularizer = 0.01 * (q - q_reference)
        return np.concatenate((scratch.body("hand").xpos - target, 0.5 * rotation, regularizer))

    result = least_squares(residual, q_reference, bounds=(lower, upper),
                           xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=500)
    return result.x


def run(seed: int = 0, settle_steps: int = 150, stage_steps: int = 600) -> dict:
    path = ROOT / "sim" / "assets" / "panda_push_grasp.xml"
    model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("task_home").id)
    rng = np.random.default_rng(seed)
    cube_xy = np.array([0.50, 0.0]) + rng.uniform(-0.005, 0.005, size=2)
    data.joint("cube_free").qpos[:2] = cube_xy
    arm_qpos, arm_dof = _arm_addresses(model)
    data.ctrl[:] = model.key("task_home").ctrl
    for _ in range(settle_steps): mujoco.mj_step(model, data)
    desired_quat = np.empty(4)
    mujoco.mju_mat2Quat(desired_quat, data.body("hand").xmat)

    initial_z = float(data.body("task_cube").xpos[2])
    contact_steps = 0; bilateral_steps = 0; max_z = initial_z
    stage_trace = []
    pose_stages = (
        # In the upstream hand frame the finger bodies are 58.4 mm below the
        # hand origin and the pads extend farther downward. A 125 mm hand
        # height centers the 50 mm cube between the physical pads.
        (np.array([cube_xy[0], cube_xy[1], 0.20]), 255.0),
        (np.array([cube_xy[0], cube_xy[1], 0.125]), 255.0),
        (np.array([cube_xy[0], cube_xy[1], 0.125]), 0.0),
        (np.array([cube_xy[0], cube_xy[1], 0.27]), 0.0),
    )
    home_q = data.qpos[arm_qpos].copy()
    ik_targets = []
    previous = home_q
    for target, gripper in pose_stages:
        solved = _solve_pose_ik(model, previous, target, desired_quat, arm_qpos)
        ik_targets.append((target, gripper, solved)); previous = solved

    for target, gripper, joint_target in ik_targets:
        stage_left = stage_right = 0
        start = data.ctrl[:7].copy()
        for step in range(stage_steps):
            blend = min(1.0, (step + 1) / max(1, int(stage_steps * 0.75)))
            data.ctrl[:7] = (1.0 - blend) * start + blend * joint_target
            data.ctrl[7] = gripper
            mujoco.mj_step(model, data)
            left, right = _finger_cube_contacts(model, data)
            stage_left += int(left); stage_right += int(right)
            contact_steps += int(left or right); bilateral_steps += int(left and right)
            max_z = max(max_z, float(data.body("task_cube").xpos[2]))
        stage_trace.append({
            "target_hand_xyz_m": target.tolist(), "gripper_command": gripper,
            "final_hand_xyz_m": data.body("hand").xpos.tolist(),
            "final_cube_xyz_m": data.body("task_cube").xpos.tolist(),
            "left_contact_steps": stage_left, "right_contact_steps": stage_right,
        })

    final_z = float(data.body("task_cube").xpos[2])
    return {
        "version": "panda_scripted_grasp_baseline_v1", "seed": seed,
        "initial_cube_xy_m": cube_xy.tolist(),
        "xml": str(path.relative_to(ROOT)), "uses_weld": False,
        "uses_handwritten_grasp_flag": False, "initial_cube_z_m": initial_z,
        "final_cube_z_m": final_z, "max_cube_z_m": max_z,
        "lift_m": max_z - initial_z, "finger_contact_steps": contact_steps,
        "bilateral_contact_steps": bilateral_steps,
        "stage_trace": stage_trace,
        "final_hand_xyz_m": data.body("hand").xpos.tolist(),
        "final_left_finger_xyz_m": data.body("left_finger").xpos.tolist(),
        "final_right_finger_xyz_m": data.body("right_finger").xpos.tolist(),
        "success": bool(max_z - initial_z >= 0.05 and bilateral_steps >= 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/panda_scripted_grasp_baseline_v1/metrics.json"))
    parser.add_argument("--stage-steps", type=int, default=600)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    rows = [run(seed=seed, stage_steps=args.stage_steps) for seed in range(args.trials)]
    result = {
        "version": "panda_scripted_grasp_baseline_v1",
        "trials": args.trials, "stage_steps": args.stage_steps,
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_lift_m": float(np.mean([row["lift_m"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
