"""Behavior-cloning sanity check for the original 5-DoF Push interface.

This is a diagnostic, not a proposed method. It asks whether contact-producing
directional demonstrations can be represented and imitated through the exact
observation/action interface used by the TD-MPC2 transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotarm.envs.damage import D2, D3, D4, DamageConfig
from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 33, action_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def damage_from_name(name: str) -> DamageConfig:
    return {
        "intact": DamageConfig.intact,
        "D2": D2,
        "D3": D3,
        "D4": D4,
    }[name]()


def collect_demonstrations(
    episodes: int, seed: int, damage: str = "intact"
):
    env = OriginalArmPushEnv(
        seed=seed,
        seed_policy="directional",
        damage=damage_from_name(damage),
    )
    observations, actions = [], []
    successes = []
    try:
        for episode in range(episodes):
            obs = env.reset(seed=seed + episode)
            info = None
            for _ in range(env.max_episode_steps):
                action = env.rand_act()
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, _, done, info = env.step(action)
                if done:
                    break
            assert info is not None
            successes.append(bool(info["success"]))
    finally:
        env.close()
    return np.stack(observations), np.stack(actions), successes


@torch.no_grad()
def evaluate(
    policy,
    obs_mean,
    obs_std,
    episodes: int,
    seed: int,
    device: str,
    damage: str = "intact",
):
    env = OriginalArmPushEnv(
        seed=seed,
        seed_policy="random",
        damage=damage_from_name(damage),
    )
    successes, distances, contacts = [], [], []
    try:
        for episode in range(episodes):
            obs = env.reset(seed=seed + episode)
            had_contact = False
            info = None
            for _ in range(env.max_episode_steps):
                x = torch.as_tensor((obs - obs_mean) / obs_std, device=device)
                action = policy(x.unsqueeze(0)).squeeze(0).cpu().numpy()
                obs, _, done, info = env.step(action)
                had_contact |= bool(info["contact"])
                if done:
                    break
            assert info is not None
            successes.append(bool(info["success"]))
            distances.append(float(info["goal_distance_m"]))
            contacts.append(had_contact)
    finally:
        env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "contact_episode_rate": float(np.mean(contacts)),
        "terminal_goal_distance_m_mean": float(np.mean(distances)),
        "terminal_goal_distance_m_median": float(np.median(distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--damage", choices=("intact", "D2", "D3", "D4"), default="intact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observations, actions, teacher_successes = collect_demonstrations(
        args.train_episodes, args.seed, args.damage
    )
    obs_mean = observations.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(observations.std(axis=0), 1e-4).astype(np.float32)
    x = torch.as_tensor((observations - obs_mean) / obs_std, device=device)
    y = torch.as_tensor(actions, device=device)
    policy = BCPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    batch_size = 256
    final_loss = float("nan")
    for _ in range(args.epochs):
        permutation = torch.randperm(len(x), device=device)
        for start in range(0, len(x), batch_size):
            index = permutation[start : start + batch_size]
            loss = torch.mean((policy(x[index]) - y[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())

    result = {
        "diagnostic": "behavior_cloning_sanity",
        "train_episodes": args.train_episodes,
        "damage": args.damage,
        "train_transitions": int(len(observations)),
        "teacher_train_success_rate": float(np.mean(teacher_successes)),
        "epochs": args.epochs,
        "final_minibatch_mse": final_loss,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.seed + 10_000,
        "evaluation": evaluate(
            policy,
            obs_mean,
            obs_std,
            args.eval_episodes,
            args.seed + 10_000,
            device,
            args.damage,
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
