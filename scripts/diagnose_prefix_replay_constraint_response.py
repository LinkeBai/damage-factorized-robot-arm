"""Exact-prefix contact response audit on the calibrated GenkiArm Push model.

Each episode is rolled to physical tool--block contact with all fault equalities
inactive.  The complete MuJoCo ``MjData`` is then copied into intact and locked
branches.  Both receive the same next action; the only intervention is an
equality lock activated at the current joint angle in the locked branch.

This avoids treating a 14-D observation as a complete simulator state and is
therefore suitable for testing contact-phase causal attribution.  Solver
reaction forces remain training/diagnostic labels, never deployment inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from scripts.diagnose_constraint_response_identifiability import inverse_mass, r2_score
from scripts.diagnose_contact_constraint_response import generalized_force_by_type


JOINTS = ("j1", "j2", "j3", "j4", "j5")
LOCKS = ("j2", "j3", "j4")
CTRL_SCALE = np.array([1.5, 1.8, 2.4, 1.8, 3.0], dtype=np.float64)
CONTACT_GEOMS = ("tool_collision", "pusher_collision")


def _has_tool_block_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    block = int(model.geom("block_geom").id)
    tools = {int(model.geom(name).id) for name in CONTACT_GEOMS}
    return any(
        block in {int(c.geom1), int(c.geom2)}
        and bool(tools.intersection({int(c.geom1), int(c.geom2)}))
        for c in data.contact
    )


def _copy_data(model: mujoco.MjModel, source: mujoco.MjData) -> mujoco.MjData:
    destination = mujoco.MjData(model)
    mujoco.mj_copyData(destination, model, source)
    return destination


def _calibrated_r2(
    actual: np.ndarray, predicted: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[float, float]:
    denominator = max(float(np.sum(predicted[train] ** 2)), 1e-12)
    alpha = float(np.sum(predicted[train] * actual[train]) / denominator)
    return alpha, r2_score(actual[test], alpha * predicted[test])


def run(xml: Path, *, seed: int, episodes: int, samples_per_episode: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    model = model_with_inactive_joint_locks(xml, JOINTS)
    joint_ids = np.array([model.joint(name).id for name in JOINTS], dtype=int)
    qpos_adrs = model.jnt_qposadr[joint_ids]
    dof_adrs = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array([model.actuator(f"m{i}").id for i in range(1, 6)], dtype=int)
    block_dofs = np.array([
        model.joint("block_x").dofadr[0], model.joint("block_y").dofadr[0]
    ], dtype=int)

    records: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    for episode in range(episodes):
        prefix = mujoco.MjData(model)
        block_xy = np.array([
            rng.uniform(0.185, 0.225), rng.uniform(0.075, 0.125)
        ])
        block_origin = np.asarray(model.body("block").pos[:2])
        prefix.qpos[[model.joint("block_x").qposadr[0], model.joint("block_y").qposadr[0]]] = (
            block_xy - block_origin
        )
        mujoco.mj_forward(model, prefix)
        approach, _ = solve_reach_reference(
            np.array([block_xy[0] - 0.03, block_xy[1], 0.025]),
            model.jnt_range[joint_ids],
        )
        push, _ = solve_reach_reference(
            np.array([block_xy[0] + 0.08, block_xy[1], 0.020]),
            model.jnt_range[joint_ids],
        )
        contacts = 0
        for step in range(450):
            state = np.concatenate([prefix.qpos[qpos_adrs], prefix.qvel[dof_adrs]])
            reference = approach if step < 160 else push
            action = joint_reference_action(state, reference)
            # Small deterministic excitation avoids a single repeated action
            # while retaining the same action in both counterfactual branches.
            if step >= 160:
                action = np.clip(
                    action + 0.04 * np.sin(0.17 * step + np.arange(5)), -1.0, 1.0
                )
            prefix.ctrl[actuator_ids] = np.clip(action * CTRL_SCALE, -1.5, 1.5)
            mujoco.mj_step(model, prefix)
            if not _has_tool_block_contact(model, prefix):
                continue
            contacts += 1
            if contacts > samples_per_episode:
                continue
            next_state = np.concatenate([prefix.qpos[qpos_adrs], prefix.qvel[dof_adrs]])
            next_action = joint_reference_action(next_state, push)
            next_action = np.clip(
                next_action + 0.04 * np.sin(0.17 * (step + 1) + np.arange(5)),
                -1.0,
                1.0,
            )
            for lock_name in LOCKS:
                intact = _copy_data(model, prefix)
                locked = _copy_data(model, prefix)
                # Equality activation is per-MjData in the supported MuJoCo API.
                intact.eq_active[:] = 0
                lock_index = JOINTS.index(lock_name)
                lock_angle = float(locked.qpos[qpos_adrs[lock_index]])
                activate_joint_lock(model, locked, lock_name, lock_angle)
                ctrl = np.clip(next_action * CTRL_SCALE, -1.5, 1.5)
                intact.ctrl[actuator_ids] = ctrl
                locked.ctrl[actuator_ids] = ctrl
                prefix_error = float(np.max(np.abs(intact.qpos - locked.qpos)))
                inv_mass = inverse_mass(model, locked)
                mujoco.mj_step(model, intact)
                mujoco.mj_step(model, locked)
                equality_force = generalized_force_by_type(model, locked, equality=True)
                contact_force_delta = (
                    generalized_force_by_type(model, locked, equality=False)
                    - generalized_force_by_type(model, intact, equality=False)
                )
                dt = float(model.opt.timestep)
                equality_delta = dt * (inv_mass @ equality_force)
                all_delta = dt * (inv_mass @ (equality_force + contact_force_delta))
                actual_delta = locked.qvel - intact.qvel
                free_dofs = np.delete(dof_adrs, lock_index)
                records.append({
                    "episode": episode,
                    "lock": lock_name,
                    "state": np.concatenate([
                        prefix.qpos[qpos_adrs], prefix.qvel[dof_adrs],
                        prefix.body("block").xpos[:2], prefix.body("block").cvel[3:5],
                    ]),
                    "action": next_action.copy(),
                    "prefix_qpos_max_difference": prefix_error,
                    "actual_full": actual_delta.copy(),
                    "equality_full": equality_delta.copy(),
                    "contact_delta_full": contact_force_delta.copy(),
                    "robot_actual": actual_delta[free_dofs],
                    "robot_equality": equality_delta[free_dofs],
                    "robot_all": all_delta[free_dofs],
                    "object_actual": actual_delta[block_dofs],
                    "object_equality": equality_delta[block_dofs],
                    "object_all": all_delta[block_dofs],
                })
        episode_summaries.append({
            "episode": episode,
            "block_initial_xy": block_xy.tolist(),
            "contact_steps_observed": contacts,
            "samples_used": min(contacts, samples_per_episode),
        })

    if not records:
        return {
            "version": "exact_prefix_contact_response_v1",
            "seed": seed,
            "decision": "invalid_no_contact_samples",
            "episodes": episode_summaries,
        }
    episode_ids = np.array([int(row["episode"]) for row in records])
    shuffled = rng.permutation(episodes)
    train_episodes = set(int(x) for x in shuffled[: max(1, int(0.7 * episodes))])
    train = np.array([int(x) in train_episodes for x in episode_ids])
    test = ~train
    if not np.any(test):
        raise ValueError("episodes must leave at least one held-out episode")

    def metric(prefix: str) -> dict[str, float]:
        actual = np.stack([np.asarray(row[f"{prefix}_actual"]) for row in records])
        equality = np.stack([np.asarray(row[f"{prefix}_equality"]) for row in records])
        all_forces = np.stack([np.asarray(row[f"{prefix}_all"]) for row in records])
        eq_alpha, eq_r2 = _calibrated_r2(actual, equality, train, test)
        all_alpha, all_r2 = _calibrated_r2(actual, all_forces, train, test)
        return {
            "equality_only_alpha": eq_alpha,
            "equality_only_test_r2": eq_r2,
            "equality_plus_contact_delta_alpha": all_alpha,
            "equality_plus_contact_delta_test_r2": all_r2,
            "actual_test_delta_rms": float(np.sqrt(np.mean(actual[test] ** 2))),
        }

    result = {
        "version": "exact_prefix_contact_response_v1",
        "model": str(xml),
        "seed": seed,
        "records": len(records),
        "train_episodes": sorted(train_episodes),
        "test_episodes": sorted(set(range(episodes)) - train_episodes),
        "max_prefix_qpos_difference": float(max(
            float(row["prefix_qpos_max_difference"]) for row in records
        )),
        "episode_summaries": episode_summaries,
        "robot_free_joint": metric("robot"),
        "object": metric("object"),
    }
    result["_records"] = records
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml", type=Path, default=Path("sim/assets/genkiarm_push.xml")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path)
    args = parser.parse_args()
    result = run(
        args.xml,
        seed=args.seed,
        episodes=args.episodes,
        samples_per_episode=args.samples_per_episode,
    )
    records = result.pop("_records", [])
    if args.dataset_output is not None and records:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_output,
            state=np.stack([np.asarray(row["state"]) for row in records]),
            action=np.stack([np.asarray(row["action"]) for row in records]),
            lock=np.asarray([LOCKS.index(str(row["lock"])) for row in records]),
            episode=np.asarray([int(row["episode"]) for row in records]),
            actual_full=np.stack([np.asarray(row["actual_full"]) for row in records]),
            equality_full=np.stack([np.asarray(row["equality_full"]) for row in records]),
            contact_delta_full=np.stack([
                np.asarray(row["contact_delta_full"]) for row in records
            ]),
        )
        result["dataset_output"] = str(args.dataset_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
