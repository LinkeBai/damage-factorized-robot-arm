"""Collect cross-arm active-contact probes and H10 counterfactual forks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from collect_cross_arm_contact_counterfactuals import (
    MAX_DOF, branch_sample, candidate_actions, copy_data, genki_prefixes,
    map_arm_action, object_robot_contact, panda_prefixes, position_jacobian,
)
from robotarm.training.variable_trajectory import observe_mujoco_nodes


PROFILES = {
    "nominal": (1.0, 1.0, 1.0),
    "high_damping": (3.0, 1.0, 1.0),
    "high_friction": (1.0, 1.8, 1.0),
    "weak_actuator": (1.0, 1.0, 0.65),
    "heldout_mixed": (2.2, 1.35, 0.8),
}


def snapshot_physics(model):
    return (model.dof_damping.copy(), model.geom_friction.copy(),
            model.actuator_gainprm.copy(), model.actuator_biasprm.copy())


def set_profile(model, snapshot, profile):
    damping, friction, gain, bias = snapshot
    ds, fs, actuator = PROFILES[profile]
    model.dof_damping[:] = damping * ds
    model.geom_friction[:] = friction
    model.geom_friction[:, 0] *= fs
    model.actuator_gainprm[:] = gain
    model.actuator_biasprm[:] = bias
    model.actuator_gainprm[:, 0] *= actuator
    model.actuator_biasprm[:, 1:3] *= actuator


def pad_nodes(value, dof):
    result = np.zeros((MAX_DOF, 2), dtype=np.float32)
    result[:dof] = value
    return result


def pad_action(value, dof):
    result = np.zeros(MAX_DOF, dtype=np.float32)
    result[:dof] = value
    return result


def active_probe(*, model, prefix, joints, actuators, object_body, object_geom,
                 robot_bodies, base_action, lock_index, steps, amplitude):
    data = copy_data(model, prefix)
    joint = model.joint(joints[lock_index]); qadr = int(joint.qposadr[0]); vadr = int(joint.dofadr[0])
    lock_angle = float(data.qpos[qadr]); actuator_id = int(model.actuator(actuators[lock_index]).id)
    dof = len(joints)
    history = {key: [] for key in ("probe_joint_state", "probe_joint_delta", "probe_action",
                                   "probe_object_pose", "probe_object_twist",
                                   "probe_object_delta", "probe_contact")}
    for step in range(steps):
        before_nodes, before_pose, before_twist = observe_mujoco_nodes(
            model, data, joint_names=joints, object_body=object_body)
        phase = 2.0 * np.pi * (step + 1) / max(steps, 1)
        offsets = amplitude * np.sin(phase + np.arange(dof) * np.pi / 3.0)
        action = np.clip(np.asarray(base_action) + offsets, -1.0, 1.0)
        map_arm_action(model, data, joints, actuators, action)
        if int(model.actuator_biastype[actuator_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
            data.ctrl[actuator_id] = lock_angle
        else:
            data.ctrl[actuator_id] = 0.0
        contact_before = object_robot_contact(model, data, object_geom, robot_bodies)
        mujoco.mj_step(model, data)
        data.qpos[qadr], data.qvel[vadr] = lock_angle, 0.0
        mujoco.mj_forward(model, data)
        after_nodes, after_pose, after_twist = observe_mujoco_nodes(
            model, data, joint_names=joints, object_body=object_body)
        history["probe_joint_state"].append(pad_nodes(before_nodes, dof))
        history["probe_joint_delta"].append(pad_nodes(after_nodes - before_nodes, dof))
        history["probe_action"].append(pad_action(action, dof))
        history["probe_object_pose"].append(before_pose.astype(np.float32))
        history["probe_object_twist"].append(before_twist.astype(np.float32))
        history["probe_object_delta"].append(np.concatenate([
            after_pose[:3] - before_pose[:3], after_twist - before_twist]).astype(np.float32))
        history["probe_contact"].append(np.asarray([
            contact_before, object_robot_contact(model, data, object_geom, robot_bodies)], dtype=np.float32))
    return data, {key: np.stack(value) for key, value in history.items()}


def run(seed, prefixes_per_robot, probe_steps, candidates, horizon, output, summary_path):
    definitions = [
      ("genkiarm", genki_prefixes(prefixes_per_robot, seed), "block", "block_geom", {"tool"},
       (1,2,3), lambda d: d.site("ee").xpos.copy(), {"site":"ee"}),
      ("panda", panda_prefixes(prefixes_per_robot, seed+99), "task_cube", "cube_geom",
       {"left_finger","right_finger","hand"}, (1,3,5),
       lambda d: d.body("hand").xpos.copy(), {"body":"hand"}),
    ]
    rng = np.random.default_rng(seed + 8801); rows = []; attempted = {"genkiarm":0,"panda":0}
    for robot, prefixes, obj_body, obj_geom, bodies, locks, ee_fn, jac_target in definitions:
        for model, prefix, joints, actuators, spec, base_action in prefixes:
            prefix_id = attempted[robot]; attempted[robot] += 1; snapshot = snapshot_physics(model)
            for profile in PROFILES:
                set_profile(model, snapshot, profile); mujoco.mj_forward(model, prefix)
                for lock in locks:
                    probe_end, history = active_probe(
                        model=model, prefix=prefix, joints=joints, actuators=actuators,
                        object_body=obj_body, object_geom=obj_geom, robot_bodies=bodies,
                        base_action=base_action, lock_index=lock, steps=probe_steps, amplitude=0.18)
                    joint_dofs = [int(model.joint(name).dofadr[0]) for name in joints]
                    jac = position_jacobian(model, probe_end, **jac_target)[:, joint_dofs]
                    for action_id, action in enumerate(candidate_actions(base_action, count=candidates, rng=rng)):
                        row = branch_sample(
                          model=model, prefix=probe_end, robot=robot, joint_names=joints,
                          actuator_names=actuators, object_body=obj_body, object_geom=obj_geom,
                          robot_bodies=bodies, action=action, lock_index=lock, axes=spec.axes,
                          origins=spec.origins, ee_position=ee_fn(probe_end),
                          ee_position_jacobian=jac, branch_horizon=horizon,
                          require_current_contact=False)
                        if row is None: continue
                        row.update(history); row.update({"profile":profile, "prefix_id":prefix_id,
                          "action_id":action_id, "physics_values":np.asarray(PROFILES[profile], dtype=np.float32)})
                        rows.append(row)
            set_profile(model, snapshot, "nominal")
    if not rows: raise RuntimeError("no active-probe rows")
    keys = [key for key in rows[0] if key not in ("robot", "profile")]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, robot=np.asarray([r["robot"] for r in rows]),
      profile=np.asarray([r["profile"] for r in rows]),
      **{key:np.asarray([r[key] for r in rows]) for key in keys})
    summary = {"version":"cross_arm_active_probe_counterfactual_v1", "seed":seed,
      "prefixes_attempted":attempted, "probe_steps":probe_steps, "candidates":candidates,
      "horizon":horizon, "profiles":list(PROFILES), "rows":len(rows),
      "rows_by_robot":{r:sum(x["robot"]==r for x in rows) for r in attempted},
      "contact_after_probe_fraction":{r:float(np.mean([x["probe_contact"][-1,1] for x in rows if x["robot"]==r])) for r in attempted},
      "max_lock_violation":float(max(abs(x["state"][int(x["lock_index"]),0]-x["angle"][int(x["lock_index"])]) for x in rows))}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20260829); parser.add_argument("--prefixes-per-robot",type=int,default=80)
    parser.add_argument("--probe-steps",type=int,default=8); parser.add_argument("--candidates",type=int,default=6)
    parser.add_argument("--horizon",type=int,default=10); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--summary",type=Path,required=True); args=parser.parse_args()
    run(args.seed,args.prefixes_per_robot,args.probe_steps,args.candidates,args.horizon,args.output,args.summary)


if __name__=="__main__": main()
