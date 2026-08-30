"""Minimal DAgger diagnostic on the original 5-DoF Push task.

This is a transfer/sanity baseline, not the proposed IPWM contribution. It
tests whether querying the geometric teacher on learner-visited states repairs
the post-contact distribution shift observed with one-shot behavior cloning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv
from run_original_arm_bc_sanity import (
    BCPolicy,
    collect_demonstrations,
    damage_from_name,
    evaluate,
)


def fit_policy(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    obs_mean = observations.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(observations.std(axis=0), 1e-4).astype(np.float32)
    x = torch.as_tensor((observations - obs_mean) / obs_std, device=device)
    y = torch.as_tensor(actions, device=device)
    policy = BCPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    final_loss = float("nan")
    for _ in range(epochs):
        permutation = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 256):
            index = permutation[start : start + 256]
            loss = torch.mean((policy(x[index]) - y[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return policy, obs_mean, obs_std, final_loss


@torch.no_grad()
def aggregate_learner_states(
    policy,
    obs_mean,
    obs_std,
    *,
    episodes: int,
    seed: int,
    beta: float,
    device: str,
    damage: str = "intact",
):
    """Execute learner/teacher mixture but always label with teacher action."""
    rng = np.random.default_rng(seed)
    env = OriginalArmPushEnv(
        seed=seed,
        seed_policy="directional",
        damage=damage_from_name(damage),
    )
    observations, labels, rollout_successes = [], [], []
    try:
        for episode in range(episodes):
            obs = env.reset(seed=seed + episode)
            info = None
            for _ in range(env.max_episode_steps):
                teacher_action = env.rand_act()
                observations.append(obs.copy())
                labels.append(teacher_action.copy())
                normalized = torch.as_tensor(
                    (obs - obs_mean) / obs_std, device=device
                )
                learner_action = (
                    policy(normalized.unsqueeze(0)).squeeze(0).cpu().numpy()
                )
                execute = teacher_action if rng.random() < beta else learner_action
                obs, _, done, info = env.step(execute)
                if done:
                    break
            assert info is not None
            rollout_successes.append(bool(info["success"]))
    finally:
        env.close()
    return (
        np.stack(observations),
        np.stack(labels),
        float(np.mean(rollout_successes)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-episodes", type=int, default=40)
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--damage", choices=("intact", "D2", "D3", "D4"), default="intact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observations, actions, teacher_successes = collect_demonstrations(
        args.initial_episodes, args.seed, args.damage
    )
    history = []
    for iteration in range(args.dagger_iterations + 1):
        policy, obs_mean, obs_std, loss = fit_policy(
            observations,
            actions,
            epochs=args.epochs,
            seed=args.seed + iteration,
            device=device,
        )
        evaluation = evaluate(
            policy,
            obs_mean,
            obs_std,
            args.eval_episodes,
            args.seed + 10_000,
            device,
            args.damage,
        )
        record = {
            "iteration": iteration,
            "aggregate_transitions": int(len(observations)),
            "final_minibatch_mse": loss,
            "evaluation": evaluation,
        }
        if iteration < args.dagger_iterations:
            new_obs, new_actions, rollout_success = aggregate_learner_states(
                policy,
                obs_mean,
                obs_std,
                episodes=args.dagger_episodes,
                seed=args.seed + 1_000 * (iteration + 1),
                beta=args.beta,
                device=device,
                damage=args.damage,
            )
            record["aggregation_rollout_success_rate"] = rollout_success
            observations = np.concatenate([observations, new_obs])
            actions = np.concatenate([actions, new_actions])
        history.append(record)

    result = {
        "diagnostic": "dagger_sanity",
        "initial_episodes": args.initial_episodes,
        "damage": args.damage,
        "initial_teacher_success_rate": float(np.mean(teacher_successes)),
        "dagger_iterations": args.dagger_iterations,
        "dagger_episodes_per_iteration": args.dagger_episodes,
        "teacher_execution_probability_beta": args.beta,
        "eval_episodes": args.eval_episodes,
        "fixed_eval_seed_start": args.seed + 10_000,
        "history": history,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
