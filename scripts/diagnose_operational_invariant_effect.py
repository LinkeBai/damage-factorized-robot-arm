"""Closed-form diagnostic for an embodiment-normalized intervention observable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from collect_cross_arm_contact_counterfactuals import map_arm_action, position_jacobian
from diagnose_active_probe_identifiability import ridge_fit_predict
from run_contact_action_effect_gate import prediction_metrics, ranking_metrics


ROOT = Path(__file__).resolve().parents[1]


def definition(robot: str):
    if robot == "genkiarm":
        return (ROOT / "sim/assets/genkiarm_push.xml", tuple(f"j{i}" for i in range(1, 6)),
                tuple(f"m{i}" for i in range(1, 6)), {"site": "ee"}, None)
    return (ROOT / "sim/assets/panda_push_grasp.xml", tuple(f"joint{i}" for i in range(1, 8)),
            tuple(f"actuator{i}" for i in range(1, 8)), {"body": "hand"}, "task_home")


def response(model, data, joint_names, actuator_names, action, lock_index, jac_target):
    map_arm_action(model, data, joint_names, actuator_names, action)
    mujoco.mj_forward(model, data)
    dofs = np.asarray([int(model.joint(name).dofadr[0]) for name in joint_names])
    jacobian = position_jacobian(model, data, **jac_target)[:, dofs]
    mass_full = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, mass_full)
    except TypeError:
        packed_mass = data.M if hasattr(data, "M") else data.qM
        mujoco.mj_fullM(model, mass_full, packed_mass)
    mass = mass_full[np.ix_(dofs, dofs)]
    torque = np.asarray(data.qfrc_actuator)[dofs]

    def reduced(keep):
        inv_mass = np.linalg.pinv(mass[np.ix_(keep, keep)], rcond=1e-8)
        jac = jacobian[:, keep]; tau = torque[keep]
        acceleration = jac @ inv_mass @ tau
        mobility = jac @ inv_mass @ jac.T
        energy = max(float(tau @ inv_mass @ tau), 1e-12)
        return acceleration, mobility, energy

    full_keep = np.arange(len(dofs)); locked_keep = full_keep[full_keep != lock_index]
    full = reduced(full_keep); locked = reduced(locked_keep)
    return full, locked


def invariant_features(data):
    robots = data["robot"].astype(str); result = np.zeros((len(robots), 16), dtype=np.float64)
    cache = {}
    for index, robot in enumerate(robots):
        if robot not in cache:
            path, joints, actuators, jac_target, home = definition(robot)
            cache[robot] = (mujoco.MjModel.from_xml_path(str(path)), joints, actuators, jac_target, home)
        model, joints, actuators, jac_target, home = cache[robot]
        mjdata = mujoco.MjData(model)
        if home is not None: mujoco.mj_resetDataKeyframe(model, mjdata, model.key(home).id)
        dof = len(joints)
        for joint_index, name in enumerate(joints):
            joint = model.joint(name); mjdata.qpos[int(joint.qposadr[0])] = data["state"][index, joint_index, 0]
            mjdata.qvel[int(joint.dofadr[0])] = data["state"][index, joint_index, 1]
        mujoco.mj_forward(model, mjdata)
        full, locked = response(model, mjdata, joints, actuators, data["action"][index, :dof],
                                int(data["lock_index"][index]), jac_target)
        relative = data["ee_object_relative"][index]
        normal = relative / max(float(np.linalg.norm(relative)), 1e-8)
        tangent = np.array([-normal[1], normal[0], 0.0]); tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
        vertical = np.array([0.0, 0.0, 1.0]); frame = np.stack([normal, tangent, vertical])

        values = []
        for acceleration, mobility, energy in (full, locked):
            components = frame @ acceleration / np.sqrt(energy)
            directional_mobility = np.diag(frame @ mobility @ frame.T).clip(1e-10)
            whitened = components / np.sqrt(directional_mobility)
            values.extend(whitened.tolist())
            values.extend(np.log(directional_mobility).tolist())
        delta = frame @ (locked[0] - full[0]) / np.sqrt(full[2])
        energy_ratio = np.log(locked[2] / full[2])
        result[index] = np.asarray(values + delta.tolist() + [energy_ratio])
    return result


def base_features(data):
    lock_depth = data["lock_index"][:, None] / np.maximum(data["dof"][:, None] - 1, 1)
    lock_angle = np.sum(data["angle"] * data["mask"], axis=1, keepdims=True)
    return np.concatenate([data["object_pose"], data["object_twist"], data["ee_object_relative"],
                           data["ee_action_delta"], data["ee_projected_action"], lock_depth, lock_angle], axis=1)


def evaluate(dataset: Path, seed: int):
    with np.load(dataset) as source: data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str); rng = np.random.default_rng(seed); train_prefix = {}
    for robot in ("genkiarm", "panda"):
        ids = rng.permutation(np.unique(data["prefix_id"][robots == robot])); train_prefix[robot] = set(map(int, ids[:int(0.7 * len(ids))]))
    prefix_train = np.asarray([int(prefix) in train_prefix[robot] for prefix, robot in zip(data["prefix_id"], robots)])
    middle = np.where(robots == "genkiarm", 2, 3); heldout = data["lock_index"] == middle
    train = prefix_train & ~heldout; test = ~prefix_train & heldout
    base = base_features(data); invariant = invariant_features(data); target = data["locked_object_step"].astype(np.float64)
    selected = {key: value for key, value in data.items()}; methods = {}
    for name, x in (("original_observables", base), ("plus_operational_invariant", np.concatenate([base, invariant], axis=1))):
        test_prediction = ridge_fit_predict(x, target, train, test)
        prediction = np.zeros_like(target); prediction[test] = test_prediction
        methods[name] = {"prediction": prediction_metrics(target, prediction, robots, test),
                         "ranking": ranking_metrics(target, prediction, selected, test)}
    baseline, candidate = methods["original_observables"], methods["plus_operational_invariant"]
    terms = {"pooled_rmse_improvement": (baseline["prediction"]["pooled"] - candidate["prediction"]["pooled"]) / baseline["prediction"]["pooled"],
             "both_robots_improve": all(candidate["prediction"][r] < baseline["prediction"][r] for r in ("genkiarm", "panda")),
             "spearman_improvement": candidate["ranking"]["mean_spearman"] - baseline["ranking"]["mean_spearman"],
             "lower_regret": candidate["ranking"]["normalized_top1_regret"] < baseline["ranking"]["normalized_top1_regret"]}
    return {"version": "operational_invariant_effect_diagnostic_v1", "seed": seed,
            "split": {"train": int(train.sum()), "test": int(test.sum()), "heldout_middle_locks": True, "grouped_by_prefix": True},
            "invariant_dimension": invariant.shape[1], "methods": methods, "gate_terms": terms}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    result = evaluate(args.dataset, args.seed); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
