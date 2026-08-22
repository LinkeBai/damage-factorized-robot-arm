"""Trajectory generation for residual-physics simulation domains."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import (
    joint_reference_action,
    solve_reach_reference,
)
from robotarm.training.sim_protocol import DomainSpec


@dataclass
class SimTrajectory:
    domain_id: str
    states: torch.Tensor  # (T + 1, 10)
    actions: torch.Tensor  # commanded actions, (T, 5)
    applied_actions: torch.Tensor  # delayed/deadband-filtered actions, (T, 5)
    contact_mask: torch.Tensor | None = None  # optional per-transition contact, (T,)
    contact_impulses: torch.Tensor | None = None  # force integral on object, (T, 2)
    table_impulses: torch.Tensor | None = None  # table force integral on object, (T, 2)
    contact_records: list[list[dict[str, object]]] | None = None
    metadata: dict[str, float | int | str] = field(default_factory=dict)


def collect_trajectory(
    domain: DomainSpec,
    *,
    steps: int,
    seed: int,
    target: np.ndarray | None = None,
) -> SimTrajectory:
    """Collect one persistently excited but bounded simulation trajectory."""
    env = MujocoArmEnv(residual_physics=domain.residual)
    target = (
        np.asarray(target, dtype=np.float64)
        if target is not None
        else np.array([0.18, 0.08, 0.25], dtype=np.float64)
    )
    observation = env.reset(target=target, damage_config=domain.damage)
    rng = np.random.default_rng(seed)
    action = np.zeros(5, dtype=np.float64)
    states = [observation["state"].copy()]
    commanded: list[np.ndarray] = []
    applied: list[np.ndarray] = []

    for _ in range(steps):
        excitation = rng.uniform(-0.45, 0.45, size=5)
        action = 0.75 * action + 0.25 * excitation
        action[domain.damage.locked] = 0.0
        result = env.step(action)
        commanded.append(action.copy())
        applied.append(env.last_applied_action)
        states.append(result["observation"]["state"].copy())

    return SimTrajectory(
        domain_id=domain.domain_id,
        states=torch.as_tensor(np.stack(states), dtype=torch.float32),
        actions=torch.as_tensor(np.stack(commanded), dtype=torch.float32),
        applied_actions=torch.as_tensor(np.stack(applied), dtype=torch.float32),
    )


def collect_controller_trajectory(
    domain: DomainSpec,
    *,
    steps: int,
    seed: int,
    target: np.ndarray,
    exploration_std: float = 0.08,
) -> SimTrajectory:
    """Collect task-relevant dynamics around a successful Reach controller."""
    env = MujocoArmEnv(residual_physics=domain.residual)
    observation = env.reset(target=target, damage_config=domain.damage)
    locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
    reference, _ = solve_reach_reference(
        target, env.joint_ranges, locked_joints=locked
    )
    rng = np.random.default_rng(seed)
    states = [observation["state"].copy()]
    commanded: list[np.ndarray] = []
    applied: list[np.ndarray] = []
    for _ in range(steps):
        action = joint_reference_action(
            observation["state"],
            reference,
            locked_joints=tuple(domain.damage.locked),
        )
        action += rng.normal(0.0, exploration_std, size=5)
        action = np.clip(action, -1.0, 1.0)
        action[domain.damage.locked] = 0.0
        result = env.step(action)
        commanded.append(action.copy())
        applied.append(env.last_applied_action)
        observation = result["observation"]
        states.append(observation["state"].copy())
    return SimTrajectory(
        domain_id=domain.domain_id,
        states=torch.as_tensor(np.stack(states), dtype=torch.float32),
        actions=torch.as_tensor(np.stack(commanded), dtype=torch.float32),
        applied_actions=torch.as_tensor(np.stack(applied), dtype=torch.float32),
    )


def collect_domains(
    domains: tuple[DomainSpec, ...],
    *,
    trajectories_per_domain: int,
    steps: int,
    seed: int,
    targets: tuple[np.ndarray, ...] | None = None,
) -> list[SimTrajectory]:
    trajectories = []
    for domain_index, domain in enumerate(domains):
        for trajectory_index in range(trajectories_per_domain):
            trajectory_seed = seed + domain_index * 1000 + trajectory_index
            target = (
                targets[(domain_index + trajectory_index) % len(targets)]
                if targets
                else None
            )
            trajectories.append(
                collect_trajectory(
                    domain,
                    steps=steps,
                    seed=trajectory_seed,
                    target=target,
                )
            )
    return trajectories


def collect_controller_domains(
    domains: tuple[DomainSpec, ...],
    *,
    trajectories_per_domain: int,
    steps: int,
    seed: int,
    targets: tuple[np.ndarray, ...],
) -> list[SimTrajectory]:
    trajectories = []
    for domain_index, domain in enumerate(domains):
        for trajectory_index in range(trajectories_per_domain):
            trajectories.append(
                collect_controller_trajectory(
                    domain,
                    steps=steps,
                    seed=seed + domain_index * 1000 + trajectory_index,
                    target=targets[(domain_index + trajectory_index) % len(targets)],
                )
            )
    return trajectories
