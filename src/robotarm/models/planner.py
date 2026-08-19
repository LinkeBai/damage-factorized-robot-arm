"""Frozen short-horizon CEM planner over the conditional world model."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from robotarm.models.world_model import WorldModel


@dataclass
class PlannerConfig:
    horizon: int = 5
    candidates: int = 128
    elites: int = 16
    iterations: int = 3
    initial_std: float = 0.45
    action_penalty: float = 0.01
    limit_penalty: float = 5.0
    seed: int = 0


def _translation(
    batch: int,
    xyz: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    transform = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)
    transform[:, :3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
    return transform


def _rotation(angle: torch.Tensor, axis: str) -> torch.Tensor:
    batch = angle.shape[0]
    transform = torch.eye(4, device=angle.device, dtype=angle.dtype).repeat(
        batch, 1, 1
    )
    c, s = torch.cos(angle), torch.sin(angle)
    if axis == "x":
        transform[:, 1, 1], transform[:, 1, 2] = c, -s
        transform[:, 2, 1], transform[:, 2, 2] = s, c
    elif axis == "y":
        transform[:, 0, 0], transform[:, 0, 2] = c, s
        transform[:, 2, 0], transform[:, 2, 2] = -s, c
    elif axis == "z":
        transform[:, 0, 0], transform[:, 0, 1] = c, -s
        transform[:, 1, 0], transform[:, 1, 1] = s, c
    else:
        raise ValueError(axis)
    return transform


def torch_forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """Batched five-joint TCP position matching ``envs/fk.py``."""
    if q.dim() != 2 or q.shape[1] != 5:
        raise ValueError(f"q must have shape (batch, 5), got {tuple(q.shape)}")
    batch, device, dtype = q.shape[0], q.device, q.dtype
    transform = _translation(batch, (0, 0, 0.120), device=device, dtype=dtype)
    operations = (
        _rotation(q[:, 0], "z"),
        _translation(batch, (0, 0, 0), device=device, dtype=dtype),
        _rotation(q[:, 1], "y"),
        _translation(batch, (0, 0, 0.110), device=device, dtype=dtype),
        _rotation(q[:, 2], "y"),
        _translation(batch, (0, 0, 0.120), device=device, dtype=dtype),
        _rotation(q[:, 3], "y"),
        _translation(batch, (0, 0, 0.060), device=device, dtype=dtype),
        _rotation(q[:, 4], "z"),
        _translation(batch, (0, -0.0132, 0.110), device=device, dtype=dtype),
        _translation(batch, (0.020, 0, 0), device=device, dtype=dtype),
    )
    for operation in operations:
        transform = transform @ operation
    return transform[:, :3, 3]


class CEMPlanner:
    """Plan without updating the world model or deployment context."""

    def __init__(self, world_model: WorldModel, cfg: PlannerConfig | None = None) -> None:
        self.world_model = world_model
        self.cfg = cfg or PlannerConfig()
        if not 0 < self.cfg.elites <= self.cfg.candidates:
            raise ValueError("elites must be in [1, candidates]")

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor,
        joint_ranges: torch.Tensor,
        *,
        locked_joints: tuple[int, ...] = (),
        nominal_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.world_model.parameters()).device
        state = state.to(device=device, dtype=torch.float32).reshape(1, -1)
        context = context.to(device=device, dtype=torch.float32).reshape(1, -1)
        target = target.to(device=device, dtype=torch.float32).reshape(1, 3)
        ranges = joint_ranges.to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(self.cfg.seed)

        action_dim = self.world_model.cfg.action_dim
        if nominal_action is None:
            mean = torch.zeros(self.cfg.horizon, action_dim, device=device)
        else:
            nominal = nominal_action.to(device=device, dtype=torch.float32).reshape(1, -1)
            mean = nominal.expand(self.cfg.horizon, -1).clone()
        std = torch.full_like(mean, self.cfg.initial_std)
        for _ in range(self.cfg.iterations):
            noise = torch.randn(
                self.cfg.candidates,
                self.cfg.horizon,
                action_dim,
                generator=generator,
                device=device,
            )
            actions = torch.clamp(
                mean.unsqueeze(0) + std.unsqueeze(0) * noise,
                -1.0,
                1.0,
            )
            if locked_joints:
                actions[:, :, list(locked_joints)] = 0.0
            simulated = state.expand(self.cfg.candidates, -1)
            context_batch = context.expand(self.cfg.candidates, -1)
            hidden = None
            for step in range(self.cfg.horizon):
                prediction, hidden = self.world_model.step(
                    simulated,
                    actions[:, step],
                    context_batch,
                    hidden,
                )
                simulated = prediction["mean"]

            qpos = simulated[:, :action_dim]
            tcp = torch_forward_kinematics(qpos)
            target_cost = torch.linalg.vector_norm(tcp - target, dim=-1)
            action_cost = self.cfg.action_penalty * actions.pow(2).mean(
                dim=(1, 2)
            )
            lower_violation = torch.relu(ranges[:, 0] - qpos)
            upper_violation = torch.relu(qpos - ranges[:, 1])
            limit_cost = self.cfg.limit_penalty * (
                lower_violation.pow(2) + upper_violation.pow(2)
            ).mean(dim=-1)
            cost = target_cost + action_cost + limit_cost
            elite_indices = torch.topk(
                cost, self.cfg.elites, largest=False
            ).indices
            elites = actions[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(0.05)

        action = mean[0].clamp(-1.0, 1.0)
        if locked_joints:
            action[list(locked_joints)] = 0.0
        return action.cpu()


class RobustPushCEMPlanner:
    """CEM Push planner using mean or worst-case ensemble dynamics cost."""

    def __init__(
        self,
        world_models: list[WorldModel],
        contexts: list[torch.Tensor],
        cfg: PlannerConfig | None = None,
        *,
        risk_alpha: float = 1.0,
    ) -> None:
        if not world_models or len(world_models) != len(contexts):
            raise ValueError("world_models and contexts must be non-empty and aligned")
        self.world_models = world_models
        self.contexts = contexts
        self.cfg = cfg or PlannerConfig()
        self.risk_alpha = risk_alpha

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        target_xy: torch.Tensor,
        joint_ranges: torch.Tensor,
        *,
        locked_joints: tuple[int, ...] = (),
        nominal_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.world_models[0].parameters()).device
        state = state.to(device=device, dtype=torch.float32).reshape(1, -1)
        target_xy = target_xy.to(device=device, dtype=torch.float32).reshape(1, 2)
        ranges = joint_ranges.to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(self.cfg.seed)
        action_dim = self.world_models[0].cfg.action_dim
        if nominal_action is None:
            mean = torch.zeros(self.cfg.horizon, action_dim, device=device)
        else:
            nominal = nominal_action.to(device=device, dtype=torch.float32).reshape(1, -1)
            mean = nominal.expand(self.cfg.horizon, -1).clone()
        std = torch.full_like(mean, self.cfg.initial_std)
        for _ in range(self.cfg.iterations):
            actions = torch.clamp(
                mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                    self.cfg.candidates, self.cfg.horizon, action_dim,
                    generator=generator, device=device,
                ), -1.0, 1.0,
            )
            if locked_joints:
                actions[:, :, list(locked_joints)] = 0.0
            member_costs = []
            for model, context in zip(self.world_models, self.contexts):
                simulated = state.expand(self.cfg.candidates, -1)
                context_batch = context.to(device).reshape(1, -1).expand(
                    self.cfg.candidates, -1
                )
                hidden = None
                for step in range(self.cfg.horizon):
                    prediction, hidden = model.step(
                        simulated, actions[:, step], context_batch, hidden
                    )
                    simulated = prediction["mean"]
                block_cost = torch.linalg.vector_norm(
                    simulated[:, 10:12] - target_xy, dim=-1
                )
                qpos = simulated[:, :action_dim]
                violation = torch.relu(ranges[:, 0] - qpos).pow(2)
                violation += torch.relu(qpos - ranges[:, 1]).pow(2)
                member_costs.append(
                    block_cost + self.cfg.limit_penalty * violation.mean(dim=-1)
                )
            stacked_cost = torch.stack(member_costs)
            average_cost = stacked_cost.mean(dim=0)
            cost = average_cost + self.risk_alpha * (
                stacked_cost.max(dim=0).values - average_cost
            )
            cost += self.cfg.action_penalty * actions.pow(2).mean(dim=(1, 2))
            elite_indices = torch.topk(cost, self.cfg.elites, largest=False).indices
            elites = actions[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(0.05)
        action = mean[0].clamp(-1.0, 1.0)
        if locked_joints:
            action[list(locked_joints)] = 0.0
        return action.cpu()
