"""Frozen-MPC Reach evaluation for conditional world models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.planner import CEMPlanner, PlannerConfig
from robotarm.models.residual_context import LatentOptConfig, compose_context, latent_optimize
from robotarm.training.g1_mechanism import (
    RESIDUAL_DIM,
    TrainedMechanismModels,
    encode_damage_batch,
)
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import DomainSpec


@dataclass
class FrozenMPCMetrics:
    success_rate: float
    mean_final_distance: float
    mean_steps: float
    episodes: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "success_rate": self.success_rate,
            "mean_final_distance": self.mean_final_distance,
            "mean_steps": self.mean_steps,
            "episodes": self.episodes,
        }


def infer_dfwm_context(
    models: TrainedMechanismModels,
    domain: DomainSpec,
    calibration: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    shots: int,
    latent_steps: int,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        topology = encode_damage_batch(
            models.dfwm_encoder,
            [domain.damage],
            joint_ranges,
            device,
        ).squeeze(0)
    if shots == 0:
        residual = torch.zeros(RESIDUAL_DIM, device=device)
    else:
        selected = calibration[:shots]
        states = torch.stack([trajectory.states for trajectory in selected]).to(device)
        actions = torch.stack([trajectory.actions for trajectory in selected]).to(device)
        residual = latent_optimize(
            models.dfwm_world_model,
            topology,
            states,
            actions,
            LatentOptConfig(d=RESIDUAL_DIM, steps=latent_steps, lr=0.1),
        ).z.detach()
    return compose_context(
        topology,
        residual,
        context_dim=models.dfwm_world_model.cfg.context_dim,
    ).detach()


def topology_only_context(
    models: TrainedMechanismModels,
    domain: DomainSpec,
    joint_ranges: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        return encode_damage_batch(
            models.topology_encoder,
            [domain.damage],
            joint_ranges,
            device,
        ).squeeze(0)


def evaluate_frozen_mpc(
    world_model,
    context: torch.Tensor,
    domain: DomainSpec,
    targets: tuple[np.ndarray, ...],
    *,
    max_steps: int,
    planner_config: PlannerConfig,
    tolerance: float = 0.05,
) -> FrozenMPCMetrics:
    env = MujocoArmEnv(residual_physics=domain.residual)
    planner = CEMPlanner(world_model, planner_config)
    ranges = torch.as_tensor(env.joint_ranges, dtype=torch.float32)
    successes = 0
    distances = []
    episode_steps = []
    for target in targets:
        observation = env.reset(target=target, damage_config=domain.damage)
        reached = False
        steps_taken = max_steps
        for step in range(max_steps):
            action = planner.plan(
                torch.as_tensor(observation["state"], dtype=torch.float32),
                context,
                torch.as_tensor(target, dtype=torch.float32),
                ranges,
                locked_joints=tuple(domain.damage.locked),
            ).numpy()
            result = env.step(action)
            observation = result["observation"]
            distance = float(np.linalg.norm(env.ee_pos() - target))
            if distance <= tolerance:
                reached = True
                steps_taken = step + 1
                break
        successes += int(reached)
        distances.append(distance)
        episode_steps.append(steps_taken)
    return FrozenMPCMetrics(
        success_rate=successes / len(targets),
        mean_final_distance=float(np.mean(distances)),
        mean_steps=float(np.mean(episode_steps)),
        episodes=len(targets),
    )
