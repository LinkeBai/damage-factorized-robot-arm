"""Generate solver-native, non-duplicated sequence candidates for the frozen protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from robotarm.envs.constraint_lock import activate_joint_lock, model_with_inactive_joint_locks


JOINTS = ("j1", "j2", "j3", "j4", "j5")
ACTUATORS = ("m1", "m2", "m3", "m4", "m5")
LOCK_INDEX = {f"D{index + 1}": index for index in range(5)}
CONTACT_QPOS = np.array([0.34396525, 0.79659639, 0.79649504, 1.42124113, 1.55083163])
GOALS = np.array([[0.28, 0.10], [0.26, 0.15], [0.24, 0.06]])


def arm_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    joint_ids = np.asarray([model.joint(name).id for name in JOINTS])
    return np.concatenate((
        data.qpos[model.jnt_qposadr[joint_ids]],
        data.qvel[model.jnt_dofadr[joint_ids]],
        data.body("block").xpos[:2],
        data.body("block").cvel[:2],
    )).copy()


def set_arm_state(model, data, qpos, qvel) -> None:
    for index, name in enumerate(JOINTS):
        joint = model.joint(name)
        data.qpos[int(joint.qposadr[0])] = qpos[index]
        data.qvel[int(joint.dofadr[0])] = qvel[index]


def clone_data(model, source):
    target = mujoco.MjData(model)
    target.qpos[:] = source.qpos
    target.qvel[:] = source.qvel
    target.act[:] = source.act
    target.ctrl[:] = source.ctrl
    target.time = source.time
    if hasattr(source, "eq_active"):
        target.eq_active[:] = source.eq_active
    mujoco.mj_forward(model, target)
    return target


def contact_geom_pairs(model) -> tuple[frozenset[int], ...]:
    """All end-effector geoms that can physically contact the task object."""
    block = int(model.geom("block_geom").id)
    return tuple(
        frozenset((int(model.geom(name).id), block))
        for name in ("tool_geom", "pusher_geom")
    )


def in_contact(model, data) -> bool:
    pairs = set(contact_geom_pairs(model))
    return any(
        frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2))) in pairs
        for index in range(data.ncon)
    )


def minimum_contact_geom_distance(model, data) -> float:
    block = int(model.geom("block_geom").id)
    return min(
        float(mujoco.mj_geomDistance(
            model, data, int(model.geom(name).id), block, 1.0, None
        ))
        for name in ("tool_geom", "pusher_geom")
    )


def apply_segment(model, data, action, steps) -> tuple[bool, float]:
    touched = in_contact(model, data)
    minimum_distance = minimum_contact_geom_distance(model, data)
    for _ in range(steps):
        for index, actuator in enumerate(ACTUATORS):
            data.ctrl[model.actuator(actuator).id] = action[index]
        mujoco.mj_step(model, data)
        touched = touched or in_contact(model, data)
        minimum_distance = min(minimum_distance, minimum_contact_geom_distance(model, data))
    return touched, minimum_distance


def parse_locks(text: str) -> tuple[int, ...]:
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(set(names).difference(LOCK_INDEX))
    if unknown:
        raise ValueError(f"unknown lock names: {unknown}")
    return tuple(LOCK_INDEX[name] for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path("sim/assets/arm_push.xml"))
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--locks", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes-per-lock", type=int, default=40)
    parser.add_argument("--replans", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--steps-per-segment", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested = args.locks or ("D2,D4" if args.phase == "development" else "D3")
    locks = parse_locks(requested)
    required = {1, 3} if args.phase == "development" else {2}
    if set(locks) != required:
        raise ValueError(
            f"{args.phase} generation is fail-closed to locks {sorted(required)}, got {sorted(set(locks))}"
        )
    if args.candidates != 128:
        raise ValueError("frozen formal generator requires exactly 128 candidates")
    if args.segments * args.steps_per_segment != 50:
        raise ValueError("frozen formal generator requires a 50-step horizon")

    rng = np.random.default_rng(args.seed)
    model = model_with_inactive_joint_locks(args.xml, JOINTS)
    rows = {key: [] for key in (
        "group", "episode", "stage", "locked_joint", "initial_state",
        "action_sequence", "goal", "segment_states", "contact_by_segment",
        "minimum_contact_distance_by_segment", "terminal_cost", "success",
    )}
    group = 0
    for locked in locks:
        for episode in range(args.episodes_per_lock):
            data = mujoco.MjData(model)
            qpos = CONTACT_QPOS + rng.normal(0.0, 0.015, 5)
            qvel = rng.normal(0.0, 0.01, 5)
            set_arm_state(model, data, qpos, qvel)
            mujoco.mj_forward(model, data)
            activate_joint_lock(model, data, JOINTS[locked], qpos[locked])
            goal = GOALS[episode % len(GOALS)]
            for stage in range(args.replans):
                initial = arm_state(model, data)
                sequences = rng.uniform(
                    -0.8, 0.8, (args.candidates, args.segments, len(JOINTS)))
                sequences[:, :, locked] = 0.0
                costs = []
                for sequence in sequences:
                    rollout = clone_data(model, data)
                    snapshots, contacts, distances = [], [], []
                    for action in sequence:
                        touched, minimum_distance = apply_segment(
                            model, rollout, action, args.steps_per_segment)
                        contacts.append(touched)
                        distances.append(minimum_distance)
                        snapshots.append(arm_state(model, rollout))
                    terminal_cost = float(np.linalg.norm(snapshots[-1][10:12] - goal))
                    costs.append(terminal_cost)
                    rows["group"].append(group)
                    rows["episode"].append(episode)
                    rows["stage"].append(stage)
                    rows["locked_joint"].append(locked)
                    rows["initial_state"].append(initial)
                    rows["action_sequence"].append(sequence)
                    rows["goal"].append(goal)
                    rows["segment_states"].append(np.stack(snapshots))
                    rows["contact_by_segment"].append(np.asarray(contacts, dtype=np.int8))
                    rows["minimum_contact_distance_by_segment"].append(
                        np.asarray(distances, dtype=np.float64)
                    )
                    rows["terminal_cost"].append(terminal_cost)
                    rows["success"].append(int(terminal_cost < 0.03))
                # This affects only which state the next group starts from; all
                # 128 candidates and their outcomes are retained.
                chosen = int(np.argmin(costs)) if rng.random() < 0.7 else int(rng.integers(128))
                apply_segment(model, data, sequences[chosen, 0], args.steps_per_segment)
                group += 1

    arrays = {key: np.asarray(values) for key, values in rows.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "phase": args.phase,
        "seed": args.seed,
        "locks": list(locks),
        "groups": group,
        "rows": int(arrays["group"].shape[0]),
        "episodes_per_lock": args.episodes_per_lock,
        "replans": args.replans,
        "candidates": args.candidates,
        "segments": args.segments,
        "steps_per_segment": args.steps_per_segment,
        "solver_native_lock": True,
        "all_candidates_retained": True,
        "contact_geoms": ["tool_geom", "pusher_geom"],
        "continuous_segment_contact_measurement": True,
        "file": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
