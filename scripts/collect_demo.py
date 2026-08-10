"""Collect simulated reach episodes into the append-only dataset.

Closure that ties together the whole offline pipeline (project-plan §4.7 / §5.1
/ §10): sample common-reachable targets -> reset ``MujocoArmEnv`` -> run a
simple scripted policy -> write each completed episode to an
:class:`~robotarm.data.storage.EpisodeDataset`.

This is a *demonstration / data-pipeline* script, not the training harness. The
policy is deliberately trivial (a straight-line joint-space interpolation toward
a nominal pose + small noise) so the dataset is well-formed and the storage + FK
+ env plumbing is exercised end to end. The real DFWM collect path
(``training/collect.py``) will replace the policy, but the episode shape,
schema and storage are exactly what it produces and consumes.

Usage::

    python scripts/collect_demo.py --out datasets/demo_v1 --n 40

Run after ``pip install -e .[dev]`` (package is importable as ``robotarm``).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from robotarm.data.schema import Episode, StepRecord
from robotarm.data.storage import EpisodeDataset
from robotarm.envs.damage import D2, D3, D4, DamageConfig, make_damage
from robotarm.envs.fk import forward_kinematics, inverse_kinematics
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.reachability import (
    analyze_damage_morphology,
    sample_targets_from_common,
)


def _joint_ranges(env: MujocoArmEnv) -> np.ndarray:
    return env.joint_ranges


def make_policy_episode(env, target: np.ndarray, damage: DamageConfig | None, seed: int) -> Episode:
    """Run one scripted episode toward ``target`` and return the Episode."""
    rng = np.random.default_rng(seed)
    obs = env.reset(target=target, damage_config=damage)
    locked = (
        {joint: damage.lock_angle_of(joint) for joint in damage.locked}
        if damage is not None
        else {}
    )
    nominal, ik_error = inverse_kinematics(
        target,
        env.joint_ranges,
        q0=env.joint_positions,
        locked=locked,
    )
    steps: list[StepRecord] = []

    max_steps = 80
    success = False
    for i in range(max_steps):
        # Joint-space proportional baseline toward a numerical IK solution.
        cmd = (nominal - env.joint_positions) * 1.5 + rng.normal(0, 0.01, 6)
        cmd = np.clip(cmd, -1.0, 1.0)
        for joint in locked:
            cmd[joint] = 0.0

        res = env.step(cmd)
        next_obs = res["observation"]
        done = bool(res["done"]) or i == max_steps - 1
        success = success or bool(res["success"])

        step = StepRecord(
            observation=obs,
            action_commanded=cmd,
            action_applied=cmd,
            next_observation=next_obs,
            reward=float(res["reward"]),
            success=success,
            done=done,
            safety_flags={"valid": True},
            hardware_state={"estep_called": False, "ik_error": ik_error},
        )
        steps.append(step)
        obs = next_obs
        if success:
            break

    ep = Episode(
        episode_id=f"demo_{seed:06d}",
        timestamp_ns=time.time_ns(),
        platform="sim",
        task_id="reach",
        target_id=f"tgt_{seed:06d}",
        split="calibration",
        damage_id="D0" if damage is None or damage.n_locked == 0 else "D_locked",
        joint_mask=(damage.joint_mask if damage else np.zeros(5, dtype=np.int64)),
        lock_angle=(damage.lock_angle if damage else np.zeros(5)),
        steps=steps,
        seed=seed,
        config_hash="demo_scripted_v1",
    )
    return ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/demo_v1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--damage", default="none", choices=["none", "D2", "D3", "D4"])
    ap.add_argument("--reachability-samples", type=int, default=20_000)
    args = ap.parse_args()

    env = MujocoArmEnv()
    ranges = _joint_ranges(env)

    morphologies = [
        ("intact", DamageConfig.intact()),
        ("D2", D2()),
        ("D3", D3()),
        ("D4", D4()),
    ]
    results = [
        analyze_damage_morphology(
            name,
            forward_kinematics,
            ranges,
            damage,
            n=args.reachability_samples,
            rng=np.random.default_rng(100 + i),
        )
        for i, (name, damage) in enumerate(morphologies)
    ]
    targets = sample_targets_from_common(results, n=args.n, voxel_size=0.03)

    ds = EpisodeDataset(root=Path(args.out), version="v1")
    for seed, t in enumerate(targets):
        dmg = None if args.damage == "none" else make_damage(args.damage)
        ep = make_policy_episode(env, t, dmg, seed)
        ds.add(ep, source="sim")
        print(f"  add {ep.episode_id}: len={ep.length} target={np.round(t,3)}")

    print(f"Dataset written to {ds.root}")
    print(f"  episodes={len(ds)}  integrity_ok={all(ds.verify_integrity().values())}")


if __name__ == "__main__":
    main()
