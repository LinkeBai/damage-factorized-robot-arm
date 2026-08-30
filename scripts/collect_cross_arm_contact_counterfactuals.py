"""Collect exact-state intact/locked contact counterfactuals on both arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from robotarm.models.variable_dof_ipwm import SerialChainSpec
from robotarm.training.variable_trajectory import observe_mujoco_nodes


ROOT = Path(__file__).resolve().parent.parent
MAX_DOF = 7


def copy_data(model: mujoco.MjModel, source: mujoco.MjData) -> mujoco.MjData:
    destination = mujoco.MjData(model)
    mujoco.mj_copyData(destination, model, source)
    return destination


def object_robot_contact(
    model: mujoco.MjModel, data: mujoco.MjData, object_geom: str, robot_bodies: set[str]
) -> bool:
    object_id = int(model.geom(object_geom).id)
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if object_id not in pair:
            continue
        other = next(iter(pair - {object_id}))
        if model.body(int(model.geom_bodyid[other])).name in robot_bodies:
            return True
    return False


def map_arm_action(
    model: mujoco.MjModel, data: mujoco.MjData, joint_names, actuator_names, action
) -> None:
    for index, (joint_name, actuator_name) in enumerate(zip(joint_names, actuator_names)):
        joint = model.joint(joint_name)
        actuator = model.actuator(actuator_name)
        actuator_id = int(actuator.id)
        low, high = np.asarray(actuator.ctrlrange)
        if int(model.actuator_biastype[actuator_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
            target = data.qpos[int(joint.qposadr[0])] + 0.05 * action[index]
            data.ctrl[actuator_id] = np.clip(target, low, high)
        else:
            data.ctrl[actuator_id] = action[index] * max(abs(float(low)), abs(float(high)))


def branch_sample(
    *, model, prefix, robot, joint_names, actuator_names, object_body, object_geom,
    robot_bodies, action, lock_index, axes, origins, ee_position,
    ee_position_jacobian, branch_horizon=1, require_current_contact=True,
):
    if require_current_contact and not object_robot_contact(model, prefix, object_geom, robot_bodies):
        return None
    before_nodes, before_pose, before_twist = observe_mujoco_nodes(
        model, prefix, joint_names=joint_names, object_body=object_body
    )
    intact, locked = copy_data(model, prefix), copy_data(model, prefix)
    map_arm_action(model, intact, joint_names, actuator_names, action)
    map_arm_action(model, locked, joint_names, actuator_names, action)
    locked_joint = model.joint(joint_names[lock_index])
    qadr, vadr = int(locked_joint.qposadr[0]), int(locked_joint.dofadr[0])
    lock_angle = float(locked.qpos[qadr])
    locked_actuator = model.actuator(actuator_names[lock_index])
    if int(model.actuator_biastype[int(locked_actuator.id)]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
        locked.ctrl[int(locked_actuator.id)] = lock_angle
    else:
        locked.ctrl[int(locked_actuator.id)] = 0.0
    prefix_difference = max(
        float(np.max(np.abs(intact.qpos - locked.qpos))),
        float(np.max(np.abs(intact.qvel - locked.qvel))),
    )
    for _ in range(branch_horizon):
        mujoco.mj_step(model, intact)
        mujoco.mj_step(model, locked)
        locked.qpos[qadr], locked.qvel[vadr] = lock_angle, 0.0
        mujoco.mj_forward(model, locked)
    intact_nodes, intact_pose, intact_twist = observe_mujoco_nodes(
        model, intact, joint_names=joint_names, object_body=object_body
    )
    locked_nodes, locked_pose, locked_twist = observe_mujoco_nodes(
        model, locked, joint_names=joint_names, object_body=object_body
    )
    dof = len(joint_names)
    padded = {}
    for key, value, width in (
        ("state", before_nodes, (MAX_DOF, 2)),
        ("joint_delta", locked_nodes - intact_nodes, (MAX_DOF, 2)),
        ("axes", axes, (MAX_DOF, 3)), ("origins", origins, (MAX_DOF, 3)),
    ):
        array = np.zeros(width, dtype=np.float32); array[:dof] = value; padded[key] = array
    padded_action = np.zeros(MAX_DOF, dtype=np.float32); padded_action[:dof] = action
    mask = np.zeros(MAX_DOF, dtype=np.float32); mask[lock_index] = 1.0
    angle = np.zeros(MAX_DOF, dtype=np.float32); angle[lock_index] = lock_angle
    object_delta = np.concatenate([
        locked_pose[:3] - intact_pose[:3], locked_twist - intact_twist
    ]).astype(np.float32)
    locked_object_step = np.concatenate([
        locked_pose[:3] - before_pose[:3], locked_twist - before_twist
    ]).astype(np.float32)
    # Deployable analytic feature: the first-order end-effector action response
    # removed by the diagnosed lock.  It uses only current kinematics, the
    # candidate action, and the lock mask--never solver force or future state.
    ee_action_delta = (
        -np.asarray(ee_position_jacobian, dtype=np.float32)[:, lock_index]
        * float(action[lock_index])
    )
    projected_action = np.asarray(action, dtype=np.float32).copy()
    projected_action[lock_index] = 0.0
    ee_projected_action = (
        np.asarray(ee_position_jacobian, dtype=np.float32) @ projected_action
    )
    return {
        "robot": robot, "dof": dof, "state": padded["state"],
        "action": padded_action, "mask": mask, "angle": angle,
        "axes": padded["axes"], "origins": padded["origins"],
        "object_pose": before_pose.astype(np.float32),
        "object_twist": before_twist.astype(np.float32),
        "ee_object_relative": (before_pose[:3] - ee_position).astype(np.float32),
        "joint_delta": padded["joint_delta"], "object_delta": object_delta,
        "locked_object_step": locked_object_step,
        "ee_action_delta": ee_action_delta.astype(np.float32),
        "ee_projected_action": ee_projected_action.astype(np.float32),
        "lock_index": lock_index, "prefix_difference": prefix_difference,
        "intact_contact_after": object_robot_contact(model, intact, object_geom, robot_bodies),
        "locked_contact_after": object_robot_contact(model, locked, object_geom, robot_bodies),
        "branch_horizon": branch_horizon,
    }


def genki_prefixes(limit: int, seed: int):
    model_path = ROOT / "sim/assets/genkiarm_push.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names = tuple(f"j{i}" for i in range(1, 6)); actuator_names = tuple(f"m{i}" for i in range(1, 6))
    spec = SerialChainSpec.from_mjcf(model_path, joint_names, name="genkiarm")
    source_states, source_actions = [], []
    for source_seed in (7, 17, 27):
        with np.load(ROOT / f"runs/ipwm_constraint_response_gate_v1/contact_prefix_seed{source_seed}.npz") as data:
            source_states.append(data["state"][::3]); source_actions.append(data["action"][::3])
    states, actions = np.concatenate(source_states), np.concatenate(source_actions)
    choice = np.random.default_rng(seed).choice(len(states), size=min(limit, len(states)), replace=False)
    for index in choice:
        data = mujoco.MjData(model); state = states[index]
        for j, name in enumerate(joint_names):
            joint = model.joint(name); data.qpos[int(joint.qposadr[0])] = state[j]; data.qvel[int(joint.dofadr[0])] = state[5+j]
        origin = np.asarray(model.body("block").pos[:2])
        for j, name in enumerate(("block_x", "block_y")):
            joint = model.joint(name); data.qpos[int(joint.qposadr[0])] = state[10+j] - origin[j]; data.qvel[int(joint.dofadr[0])] = state[12+j]
        mujoco.mj_forward(model, data)
        yield model, data, joint_names, actuator_names, spec, actions[index]


def panda_prefixes(limit: int, seed: int):
    model_path = ROOT / "sim/assets/panda_push_grasp.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names = tuple(f"joint{i}" for i in range(1, 8)); actuator_names = tuple(f"actuator{i}" for i in range(1, 8))
    spec = SerialChainSpec.from_mjcf(model_path, joint_names, name="panda")
    rng = np.random.default_rng(seed)
    for _ in range(limit):
        data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for name in joint_names:
            joint = model.joint(name); qadr = int(joint.qposadr[0]); low, high = np.asarray(joint.range)
            data.qpos[qadr] = np.clip(data.qpos[qadr] + rng.uniform(-0.12, 0.12), low + 0.02, high - 0.02)
        for name in ("finger_joint1", "finger_joint2"):
            data.qpos[int(model.joint(name).qposadr[0])] = 0.0245
        mujoco.mj_forward(model, data)
        hand = data.body("hand"); cube_pos = hand.xpos + hand.xmat.reshape(3, 3) @ np.array([0.0, 0.0, 0.103])
        free = int(model.joint("cube_free").qposadr[0]); data.qpos[free:free+3] = cube_pos; data.qpos[free+3:free+7] = hand.xquat
        data.ctrl[model.actuator("actuator8").id] = 0.0245 / 0.04 * 255.0
        mujoco.mj_forward(model, data)
        yield model, data, joint_names, actuator_names, spec, rng.uniform(-0.45, 0.45, 7)


def position_jacobian(model: mujoco.MjModel, data: mujoco.MjData, *, site=None, body=None):
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    if site is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, int(model.site(site).id))
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, int(model.body(body).id))
    return jacp


def candidate_actions(base, *, count: int, rng: np.random.Generator):
    base = np.asarray(base, dtype=np.float64)
    if count < 1:
        raise ValueError("actions_per_prefix must be positive")
    yield np.clip(base, -1.0, 1.0)
    for _ in range(count - 1):
        yield np.clip(base + rng.normal(0.0, 0.35, size=base.shape), -1.0, 1.0)


def run(seed: int, prefixes_per_robot: int, output: Path, summary_path: Path,
        actions_per_prefix: int = 1, branch_horizon: int = 1):
    rows = []
    definitions = [
        ("genkiarm", genki_prefixes(prefixes_per_robot, seed), "block", "block_geom", {"tool"}, (1,2,3), lambda d: d.site("ee").xpos.copy(), {"site": "ee"}),
        ("panda", panda_prefixes(prefixes_per_robot, seed+99), "task_cube", "cube_geom", {"left_finger","right_finger","hand"}, (1,3,5), lambda d: d.body("hand").xpos.copy(), {"body": "hand"}),
    ]
    rng = np.random.default_rng(seed + 1701)
    attempted = {"genkiarm": 0, "panda": 0}; rejected = {"genkiarm": 0, "panda": 0}
    for robot, prefixes, object_body, object_geom, bodies, locks, ee_fn, jac_target in definitions:
        for model, prefix, joints, actuators, spec, action in prefixes:
            attempted[robot] += 1
            accepted_here = 0
            joint_dofs = [int(model.joint(name).dofadr[0]) for name in joints]
            jac = position_jacobian(model, prefix, **jac_target)[:, joint_dofs]
            for action_id, candidate in enumerate(candidate_actions(action, count=actions_per_prefix, rng=rng)):
                for lock in locks:
                    row = branch_sample(
                        model=model, prefix=prefix, robot=robot, joint_names=joints,
                        actuator_names=actuators, object_body=object_body, object_geom=object_geom,
                        robot_bodies=bodies, action=candidate, lock_index=lock,
                        axes=spec.axes, origins=spec.origins, ee_position=ee_fn(prefix),
                        ee_position_jacobian=jac,
                        branch_horizon=branch_horizon,
                    )
                    if row is not None:
                        row["prefix_id"] = attempted[robot] - 1
                        row["action_id"] = action_id
                        rows.append(row); accepted_here += 1
            if accepted_here == 0: rejected[robot] += 1
    if not rows:
        raise RuntimeError("no valid contact counterfactual rows")
    keys = [key for key in rows[0] if key != "robot"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, robot=np.asarray([r["robot"] for r in rows]), **{
        key: np.asarray([r[key] for r in rows]) for key in keys
    })
    summary = {
        "version": "cross_arm_contact_counterfactual_dataset_v2", "seed": seed,
        "actions_per_prefix": actions_per_prefix,
        "branch_horizon": branch_horizon,
        "prefixes_attempted": attempted, "prefixes_rejected_without_contact": rejected,
        "rows": len(rows), "rows_by_robot": {robot: sum(r["robot"] == robot for r in rows) for robot in attempted},
        "max_prefix_difference": max(float(r["prefix_difference"]) for r in rows),
        "current_contact_required": True,
        "post_step_contact_fraction": {
            robot: float(np.mean([r["intact_contact_after"] and r["locked_contact_after"] for r in rows if r["robot"] == robot]))
            for robot in attempted
        },
        "object_delta_rms": {
            robot: float(np.sqrt(np.mean(np.stack([r["object_delta"] for r in rows if r["robot"] == robot]) ** 2)))
            for robot in attempted
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--prefixes-per-robot", type=int, default=80)
    parser.add_argument("--actions-per-prefix", type=int, default=1)
    parser.add_argument("--branch-horizon", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(); run(
        args.seed, args.prefixes_per_robot, args.output, args.summary,
        actions_per_prefix=args.actions_per_prefix,
        branch_horizon=args.branch_horizon,
    )


if __name__ == "__main__": main()
