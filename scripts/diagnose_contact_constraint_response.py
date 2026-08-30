"""Contact-phase identifiability audit using frozen v4 calibration states.

The same observable contact state and first candidate action are stepped under
an intact and a solver-native D3 lock.  Equality rows are separated from other
MuJoCo constraint rows, allowing a direct test of whether lock reaction alone
explains free-joint and object counterfactual response after contact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from scripts.diagnose_constraint_response_identifiability import inverse_mass, r2_score


JOINTS = ("j1", "j2", "j3", "j4", "j5")
CTRL_SCALE = np.array([1.5, 1.8, 2.4, 1.8, 3.0], dtype=np.float64)


def generalized_force_by_type(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    equality: bool,
) -> np.ndarray:
    if data.nefc == 0:
        return np.zeros(model.nv, dtype=np.float64)
    jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
    types = np.asarray(data.efc_type[: data.nefc])
    equality_rows = types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = equality_rows if equality else ~equality_rows
    if not np.any(rows):
        return np.zeros(model.nv, dtype=np.float64)
    return jacobian[rows].T @ np.asarray(data.efc_force[: data.nefc])[rows]


def set_observable_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    for index, name in enumerate(JOINTS):
        joint = model.joint(name)
        data.qpos[int(joint.qposadr[0])] = state[index]
        data.qvel[int(joint.dofadr[0])] = state[5 + index]
    block_origin = np.asarray(model.body("block").pos[:2], dtype=np.float64)
    for offset, name in enumerate(("block_x", "block_y")):
        joint = model.joint(name)
        # Observation stores block world xy; slide qpos is relative to the
        # body's MJCF origin.
        data.qpos[int(joint.qposadr[0])] = state[10 + offset] - block_origin[offset]
        data.qvel[int(joint.dofadr[0])] = state[12 + offset]
    mujoco.mj_forward(model, data)


def tool_block_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    tool = int(model.geom("tool_geom").id)
    block = int(model.geom("block_geom").id)
    return any({int(c.geom1), int(c.geom2)} == {tool, block} for c in data.contact)


def calibrated_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[float, float]:
    alpha = float(
        np.sum(predicted[train] * actual[train])
        / max(float(np.sum(predicted[train] ** 2)), 1e-12)
    )
    return alpha, r2_score(actual[test], alpha * predicted[test])


def run(
    xml: Path,
    inputs: list[Path],
    seed: int,
    damping_scale: float,
    friction_scale: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["candidate_rows"])
    model = model_with_inactive_joint_locks(xml, JOINTS)
    controlled_dofs = np.array([model.joint(name).dofadr[0] for name in JOINTS], dtype=int)
    model.dof_damping[controlled_dofs] *= damping_scale
    model.dof_frictionloss[controlled_dofs] *= friction_scale
    free_dofs = controlled_dofs[[0, 1, 3, 4]]
    block_dofs = np.array([
        model.joint("block_x").dofadr[0], model.joint("block_y").dofadr[0]
    ], dtype=int)
    actuator_ids = np.array([model.actuator(f"m{i + 1}").id for i in range(5)], dtype=int)

    robot_actual, robot_eq, robot_all = [], [], []
    object_actual, object_eq, object_all = [], [], []
    target_ids, trigger_contact_flags = [], []
    minimum_ee_block_distances: list[float] = []
    skipped_without_contact = 0
    for row in rows:
        state = np.asarray(row["state"], dtype=np.float64)
        sequence = np.asarray(row["candidate_sequence"], dtype=np.float64).reshape(-1, 5)
        # The v4 trigger is a snapshot after a step that had contact.  The
        # post-projection/current geometry can already be separated, so roll a
        # solver-locked branch until contact is present in the current state.
        contact_rollout = mujoco.MjData(model)
        set_observable_state(model, contact_rollout, state)
        trigger_contact_flags.append(tool_block_contact(model, contact_rollout))
        activate_joint_lock(model, contact_rollout, "j3", float(state[2]))
        contact_index = None
        min_distance = float(np.linalg.norm(
            contact_rollout.site("ee").xpos - contact_rollout.body("block").xpos
        ))
        for index, candidate_action in enumerate(sequence):
            contact_rollout.ctrl[actuator_ids] = np.clip(
                candidate_action * CTRL_SCALE, -1.5, 1.5
            )
            mujoco.mj_step(model, contact_rollout)
            min_distance = min(min_distance, float(np.linalg.norm(
                contact_rollout.site("ee").xpos - contact_rollout.body("block").xpos
            )))
            if tool_block_contact(model, contact_rollout):
                contact_index = index
                break
        # If the frozen H10 candidate does not itself reach current contact,
        # continue with a deterministic IK push solely to obtain a contact
        # state for the mechanism audit.  These steps are not controller data.
        if contact_index is None:
            block_xy = np.asarray(contact_rollout.body("block").xpos[:2]).copy()
            reference, _ = solve_reach_reference(
                np.array([block_xy[0] + 0.025, block_xy[1], 0.025]),
                model.jnt_range[[model.joint(name).id for name in JOINTS]],
                locked_joints={2: float(state[2])},
            )
            for offset in range(100):
                robot_state = np.concatenate([
                    [contact_rollout.qpos[int(model.joint(name).qposadr[0])] for name in JOINTS],
                    [contact_rollout.qvel[int(model.joint(name).dofadr[0])] for name in JOINTS],
                ])
                candidate_action = joint_reference_action(
                    robot_state, reference, locked_joints=(2,)
                )
                contact_rollout.ctrl[actuator_ids] = np.clip(
                    candidate_action * CTRL_SCALE, -1.5, 1.5
                )
                mujoco.mj_step(model, contact_rollout)
                min_distance = min(min_distance, float(np.linalg.norm(
                    contact_rollout.site("ee").xpos - contact_rollout.body("block").xpos
                )))
                if tool_block_contact(model, contact_rollout):
                    contact_index = len(sequence) + offset
                    action = candidate_action
                    break
        if contact_index is None:
            skipped_without_contact += 1
            minimum_ee_block_distances.append(min_distance)
            continue
        minimum_ee_block_distances.append(min_distance)
        if contact_index < len(sequence):
            action = sequence[min(contact_index + 1, len(sequence) - 1)]
        intact, locked = mujoco.MjData(model), mujoco.MjData(model)
        intact.qpos[:] = contact_rollout.qpos
        intact.qvel[:] = contact_rollout.qvel
        locked.qpos[:] = contact_rollout.qpos
        locked.qvel[:] = contact_rollout.qvel
        mujoco.mj_forward(model, intact)
        mujoco.mj_forward(model, locked)
        ctrl = np.clip(action * CTRL_SCALE, -1.5, 1.5)
        intact.ctrl[actuator_ids] = ctrl
        locked.ctrl[actuator_ids] = ctrl
        activate_joint_lock(model, locked, "j3", float(state[2]))
        inv_mass = inverse_mass(model, locked)
        mujoco.mj_step(model, intact)
        mujoco.mj_step(model, locked)
        eq_force = generalized_force_by_type(model, locked, equality=True)
        contact_delta = (
            generalized_force_by_type(model, locked, equality=False)
            - generalized_force_by_type(model, intact, equality=False)
        )
        eq_delta = model.opt.timestep * (inv_mass @ eq_force)
        all_delta = model.opt.timestep * (inv_mass @ (eq_force + contact_delta))
        actual_delta = locked.qvel - intact.qvel
        robot_actual.append(actual_delta[free_dofs])
        robot_eq.append(eq_delta[free_dofs])
        robot_all.append(all_delta[free_dofs])
        object_actual.append(actual_delta[block_dofs])
        object_eq.append(eq_delta[block_dofs])
        object_all.append(all_delta[block_dofs])
        target_ids.append(str(row["base_target"]))

    if not robot_actual:
        return {
            "version": "contact_constraint_response_v1",
            "seed": seed,
            "source_rows": len(rows),
            "contact_rows": 0,
            "skipped_without_contact": skipped_without_contact,
            "trigger_state_current_contact_fraction": float(np.mean(trigger_contact_flags)),
            "minimum_ee_block_distance_m": float(np.min(minimum_ee_block_distances)),
            "median_minimum_ee_block_distance_m": float(np.median(minimum_ee_block_distances)),
            "decision": "invalid_contact_sample_no_go",
            "damping_scale": damping_scale,
            "friction_scale": friction_scale,
        }
    arrays = [np.stack(x) for x in (
        robot_actual, robot_eq, robot_all, object_actual, object_eq, object_all
    )]
    robot_actual_a, robot_eq_a, robot_all_a, object_actual_a, object_eq_a, object_all_a = arrays
    unique_targets = sorted(set(target_ids))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_targets)[rng.permutation(len(unique_targets))]
    train_targets = set(shuffled[: max(1, int(0.7 * len(shuffled)))])
    train = np.array([target in train_targets for target in target_ids])
    test = ~train

    robot_eq_alpha, robot_eq_r2 = calibrated_r2(robot_actual_a, robot_eq_a, train, test)
    robot_all_alpha, robot_all_r2 = calibrated_r2(robot_actual_a, robot_all_a, train, test)
    object_eq_alpha, object_eq_r2 = calibrated_r2(object_actual_a, object_eq_a, train, test)
    object_all_alpha, object_all_r2 = calibrated_r2(object_actual_a, object_all_a, train, test)
    return {
        "version": "contact_constraint_response_v1",
        "seed": seed,
        "damping_scale": damping_scale,
        "friction_scale": friction_scale,
        "source_rows": len(rows),
        "contact_rows": len(robot_actual),
        "skipped_without_contact": skipped_without_contact,
        "unique_targets": len(unique_targets),
        "train_targets": sorted(train_targets),
        "test_targets": sorted(set(unique_targets) - train_targets),
        "trigger_state_current_contact_fraction": float(np.mean(trigger_contact_flags)),
        "robot_free_joint": {
            "equality_only_alpha": robot_eq_alpha,
            "equality_only_test_r2": robot_eq_r2,
            "equality_plus_contact_delta_alpha": robot_all_alpha,
            "equality_plus_contact_delta_test_r2": robot_all_r2,
            "actual_delta_rms": float(np.sqrt(np.mean(robot_actual_a[test] ** 2))),
        },
        "object": {
            "equality_only_alpha": object_eq_alpha,
            "equality_only_test_r2": object_eq_r2,
            "equality_plus_contact_delta_alpha": object_all_alpha,
            "equality_plus_contact_delta_test_r2": object_all_r2,
            "actual_delta_rms": float(np.sqrt(np.mean(object_actual_a[test] ** 2))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path("sim/assets/arm_push.xml"))
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--damping-scale", type=float, default=2.0)
    parser.add_argument("--friction-scale", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.xml, args.input, args.seed, args.damping_scale, args.friction_scale
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
