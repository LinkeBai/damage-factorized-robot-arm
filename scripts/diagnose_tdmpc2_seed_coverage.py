"""Measure whether TD-MPC2's random seed data ever reaches Push contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv
from robotarm.envs.damage import D2, D3, D4, DamageConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=("random", "directional", "directional_unaware"),
        default="random",
    )
    parser.add_argument("--damage", choices=("intact", "D2", "D3", "D4"), default="intact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    damage = {
        "intact": DamageConfig.intact,
        "D2": D2,
        "D3": D3,
        "D4": D4,
    }[args.damage]()
    env = OriginalArmPushEnv(
        seed=args.seed, seed_policy=args.policy, damage=damage
    )
    contact_episodes = 0
    max_displacements = []
    terminal_distances = []
    successes = []
    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        had_contact = False
        max_displacement = 0.0
        info = None
        for _ in range(env.max_episode_steps):
            action = env.rand_act()
            _, _, done, info = env.step(action)
            had_contact |= bool(info["contact"])
            max_displacement = max(max_displacement, info["block_displacement_m"])
            if done:
                break
        assert info is not None
        contact_episodes += int(had_contact)
        max_displacements.append(max_displacement)
        terminal_distances.append(info["goal_distance_m"])
        successes.append(bool(info["success"]))

    result = {
        "episodes": args.episodes,
        "seed": args.seed,
        "policy": args.policy,
        "damage": args.damage,
        "contact_episode_rate": contact_episodes / args.episodes,
        "success_rate": float(np.mean(successes)),
        "max_block_displacement_m": {
            "mean": float(np.mean(max_displacements)),
            "max": float(np.max(max_displacements)),
        },
        "terminal_goal_distance_m": {
            "mean": float(np.mean(terminal_distances)),
            "median": float(np.median(terminal_distances)),
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
