"""Few-shot hard-constrained action-repair diagnostic.

HCAR learns only the residual from a fault-unaware intact reference action to
the constrained fault-aware action. The failed coordinate is zeroed after the
network, so the learned model cannot violate the known hard lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv
from run_original_arm_bc_sanity import damage_from_name


class ActionRepair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(38, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def collect_pairs(episodes: int, seed: int, damage: str):
    env = OriginalArmPushEnv(
        seed=seed,
        seed_policy="directional",
        damage=damage_from_name(damage),
    )
    obs_all, base_all, target_all, successes = [], [], [], []
    try:
        for episode in range(episodes):
            obs = env.reset(seed=seed + episode)
            info = None
            for _ in range(env.max_episode_steps):
                base = env.directional_action(fault_aware=False)
                target = env.directional_action(fault_aware=True)
                obs_all.append(obs.copy())
                base_all.append(base.copy())
                target_all.append(target.copy())
                obs, _, done, info = env.step(target)
                if done:
                    break
            assert info is not None
            successes.append(bool(info["success"]))
    finally:
        env.close()
    return (
        np.stack(obs_all).astype(np.float32),
        np.stack(base_all).astype(np.float32),
        np.stack(target_all).astype(np.float32),
        successes,
    )


@torch.no_grad()
def evaluate(model, mean, std, episodes, seed, damage, device):
    env = OriginalArmPushEnv(
        seed=seed,
        seed_policy="directional_unaware",
        damage=damage_from_name(damage),
    )
    successes, contacts, distances, violations = [], [], [], []
    try:
        locked = tuple(env.damage.locked)
        for episode in range(episodes):
            obs = env.reset(seed=seed + episode)
            info = None
            had_contact = False
            max_violation = 0.0
            for _ in range(env.max_episode_steps):
                base = env.directional_action(fault_aware=False)
                feature = np.concatenate([obs, base]).astype(np.float32)
                x = torch.as_tensor((feature - mean) / std, device=device)
                delta = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
                action = np.clip(base + delta, -1.0, 1.0)
                action[list(locked)] = 0.0
                if locked:
                    max_violation = max(
                        max_violation, float(np.max(np.abs(action[list(locked)])))
                    )
                obs, _, done, info = env.step(action)
                had_contact |= bool(info["contact"])
                if done:
                    break
            assert info is not None
            successes.append(bool(info["success"]))
            contacts.append(had_contact)
            distances.append(float(info["goal_distance_m"]))
            violations.append(max_violation)
    finally:
        env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "contact_episode_rate": float(np.mean(contacts)),
        "terminal_goal_distance_m_mean": float(np.mean(distances)),
        "terminal_goal_distance_m_median": float(np.median(distances)),
        "max_failed_action_abs": float(np.max(violations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--damage", choices=("D2", "D3", "D4"), default="D3")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, base, target, teacher_success = collect_pairs(
        args.shots, args.seed, args.damage
    )
    features = np.concatenate([obs, base], axis=1)
    labels = target - base
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
    x = torch.as_tensor((features - mean) / std, device=device)
    y = torch.as_tensor(labels, device=device)
    model = ActionRepair().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    final_loss = float("nan")
    for _ in range(args.epochs):
        permutation = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 256):
            index = permutation[start : start + 256]
            loss = torch.mean((model(x[index]) - y[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())

    result = {
        "diagnostic": "hcar_fewshot",
        "damage": args.damage,
        "shots": args.shots,
        "train_transitions": int(len(features)),
        "teacher_train_success_rate": float(np.mean(teacher_success)),
        "epochs": args.epochs,
        "final_minibatch_residual_mse": final_loss,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.seed + 10_000,
        "evaluation": evaluate(
            model,
            mean,
            std,
            args.eval_episodes,
            args.seed + 10_000,
            args.damage,
            device,
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
