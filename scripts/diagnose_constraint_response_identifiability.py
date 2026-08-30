"""Pre-contact identifiability gate for solver-native joint-lock response.

For identical state/action pairs, compare one MuJoCo step with and without a
solver-visible joint equality.  The diagnostic asks whether the generalized
force contributed by that equality predicts the counterfactual velocity delta
on free joints.  It is development-only and does not train a world model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from robotarm.envs.constraint_lock import (
    activate_joint_lock,
    model_with_inactive_joint_locks,
)


JOINTS = ("j1", "j2", "j3", "j4", "j5")


def equality_generalized_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return only the generalized force due to active equality rows."""
    if data.nefc == 0:
        return np.zeros(model.nv, dtype=np.float64)
    jacobian = np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc, model.nv)
    types = np.asarray(data.efc_type[: data.nefc])
    equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = types == equality_type
    if not np.any(rows):
        return np.zeros(model.nv, dtype=np.float64)
    return jacobian[rows].T @ np.asarray(data.efc_force[: data.nefc])[rows]


def inverse_mass(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    try:
        # MuJoCo 3.12+ binding.
        mujoco.mj_fullM(model, data, mass)
    except TypeError:
        # Older Python binding.
        packed_mass = data.M if hasattr(data, "M") else data.qM
        mujoco.mj_fullM(model, mass, packed_mass)
    return np.linalg.inv(mass)


def copy_state(source: mujoco.MjData, target: mujoco.MjData) -> None:
    target.qpos[:] = source.qpos
    target.qvel[:] = source.qvel
    target.act[:] = source.act
    target.ctrl[:] = source.ctrl
    target.time = source.time


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    centered = float(np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - residual / max(centered, 1e-12)


def run(xml_path: Path, samples_per_lock: int, seed: int) -> dict[str, object]:
    model = model_with_inactive_joint_locks(xml_path, JOINTS)
    rng = np.random.default_rng(seed)
    actual_rows: list[np.ndarray] = []
    predicted_rows: list[np.ndarray] = []
    lock_rows: list[str] = []
    controlled_dofs = np.array([model.joint(name).dofadr[0] for name in JOINTS], dtype=int)
    actuator_ids = np.array([model.actuator(f"m{i + 1}").id for i in range(5)], dtype=int)

    for lock_name in ("j2", "j3", "j4"):
        lock_local_index = JOINTS.index(lock_name)
        lock_dof = controlled_dofs[lock_local_index]
        free_dofs = controlled_dofs[controlled_dofs != lock_dof]
        for _ in range(samples_per_lock):
            intact = mujoco.MjData(model)
            locked = mujoco.MjData(model)
            qpos = np.zeros(model.nq, dtype=np.float64)
            qvel = np.zeros(model.nv, dtype=np.float64)
            for name in JOINTS:
                joint = model.joint(name)
                qadr = int(joint.qposadr[0])
                qpos[qadr] = rng.uniform(max(float(joint.range[0]), -0.7), min(float(joint.range[1]), 0.7))
                qvel[int(joint.dofadr[0])] = rng.uniform(-0.25, 0.25)
            # Keep the block away from the tool: this phase isolates lock response.
            qpos[int(model.joint("block_x").qposadr[0])] = 0.40
            qpos[int(model.joint("block_y").qposadr[0])] = 0.40
            intact.qpos[:] = qpos
            intact.qvel[:] = qvel
            intact.ctrl[actuator_ids] = rng.uniform(-1.0, 1.0, size=5)
            mujoco.mj_forward(model, intact)
            copy_state(intact, locked)
            lock_angle = float(qpos[int(model.joint(lock_name).qposadr[0])])
            activate_joint_lock(model, locked, lock_name, lock_angle)

            # Store the pre-step mass matrix; both branches have identical state.
            inv_mass = inverse_mass(model, locked)
            mujoco.mj_step(model, intact)
            mujoco.mj_step(model, locked)
            equality_force = equality_generalized_force(model, locked)
            predicted_delta = model.opt.timestep * (inv_mass @ equality_force)
            actual_delta = locked.qvel - intact.qvel
            actual_rows.append(actual_delta[free_dofs])
            predicted_rows.append(predicted_delta[free_dofs])
            lock_rows.append(lock_name)

    actual = np.stack(actual_rows)
    predicted = np.stack(predicted_rows)
    indices = rng.permutation(len(actual))
    split = int(0.7 * len(indices))
    train, test = indices[:split], indices[split:]
    denominator = float(np.sum(predicted[train] ** 2))
    alpha = float(np.sum(predicted[train] * actual[train]) / max(denominator, 1e-12))
    result: dict[str, object] = {
        "version": "constraint_response_identifiability_v1",
        "phase": "pre_contact_one_step",
        "seed": seed,
        "samples_per_lock": samples_per_lock,
        "locks": ["j2", "j3", "j4"],
        "total_samples": len(actual),
        "raw_test_r2": r2_score(actual[test], predicted[test]),
        "scalar_calibration_train_only": alpha,
        "calibrated_test_r2": r2_score(actual[test], alpha * predicted[test]),
        "test_rmse": float(np.sqrt(np.mean((actual[test] - alpha * predicted[test]) ** 2))),
        "actual_delta_rms": float(np.sqrt(np.mean(actual[test] ** 2))),
        "per_lock_calibrated_test_r2": {},
    }
    test_locks = np.asarray(lock_rows)[test]
    for name in ("j2", "j3", "j4"):
        selected = test_locks == name
        result["per_lock_calibrated_test_r2"][name] = r2_score(
            actual[test][selected], alpha * predicted[test][selected]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path("sim/assets/arm_push.xml"))
    parser.add_argument("--samples-per-lock", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.xml, args.samples_per_lock, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
