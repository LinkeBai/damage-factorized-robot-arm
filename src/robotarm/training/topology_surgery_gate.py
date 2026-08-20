"""Training and evaluation utilities for the topology-surgery Gate A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.constraint_reaction_world_model import ConstraintReactionWorldModel
from robotarm.models.gated_reaction_graph import GatedReactionGraph
from robotarm.models.reduced_coordinate_graph import ReducedCoordinateGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.models.unconstrained_residual_graph import UnconstrainedResidualGraph
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.g1_mechanism import TOPOLOGY_DIM, encode_damage_batch
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import DomainSpec, damage_from_name


Method = Literal["ordinary", "soft_topology", "topology_surgery"]


@dataclass
class SurgeryGateModel:
    encoder: TopologyEncoder
    world_model: WorldModel
    method: Method


def _damage_tensors(damages: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = torch.as_tensor(
        np.stack([damage.joint_mask for damage in damages]),
        dtype=torch.float32,
        device=device,
    )
    angles = torch.as_tensor(
        np.stack([damage.lock_angle for damage in damages]),
        dtype=torch.float32,
        device=device,
    )
    return masks, angles


def _conditioning_damages(damages: list, method: Method) -> list:
    if method == "soft_topology":
        return damages
    return [damage_from_name("intact") for _ in damages]


def _project_prediction(
    prediction: dict[str, torch.Tensor], surgery: TopologySurgery,
    mask: torch.Tensor, angle: torch.Tensor,
) -> dict[str, torch.Tensor]:
    projected = dict(prediction)
    projected["mean"] = surgery.project_state(prediction["mean"], mask, angle)
    return projected


def _training_loss(
    model: SurgeryGateModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    mask: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    surgery = TopologySurgery()
    hidden = None
    losses = []
    use_surgery = model.method == "topology_surgery"
    for step in range(actions.shape[1]):
        state = states[:, step]
        action = actions[:, step]
        target = states[:, step + 1]
        if use_surgery:
            state = surgery.project_state(state, mask, angle)
            action = surgery.project_action(action, mask)
            target = surgery.project_state(target, mask, angle)
        posterior, prior, hidden = model.world_model.observe_step(
            state, action, target, context, hidden
        )
        if use_surgery:
            posterior = _project_prediction(posterior, surgery, mask, angle)
            prior = _project_prediction(prior, surgery, mask, angle)
        losses.append(
            model.world_model.nll(posterior, target)
            + 0.5 * model.world_model.nll(prior, target)
            + 0.01 * posterior["kl"]
        )

    one_step = torch.stack(losses, dim=1).mean()
    rollout_losses = []
    horizon = min(5, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predicted = states[:, start]
        rollout_hidden = None
        for offset in range(horizon):
            action = actions[:, start + offset]
            if use_surgery:
                predicted = surgery.project_state(predicted, mask, angle)
                action = surgery.project_action(action, mask)
            output, rollout_hidden = model.world_model.step(
                predicted, action, context, rollout_hidden
            )
            predicted = output["mean"]
            if use_surgery:
                predicted = surgery.project_state(predicted, mask, angle)
            rollout_losses.append(F.mse_loss(predicted, states[:, start + offset + 1]))
    return one_step + 0.5 * torch.stack(rollout_losses).mean()


def train_surgery_gate_model(
    trajectories: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    method: Method,
    epochs: int,
    device: torch.device,
    seed: int,
) -> SurgeryGateModel:
    torch.manual_seed(seed)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    encoder = TopologyEncoder().to(device)
    world_model = WorldModel(WorldModelConfig(
        state_dim=states.shape[-1], action_dim=actions.shape[-1],
        context_dim=TOPOLOGY_DIM, latent_dim=128,
    )).to(device)
    model = SurgeryGateModel(encoder=encoder, world_model=world_model, method=method)
    params = list(encoder.parameters()) + list(world_model.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-3)
    conditioning = _conditioning_damages(damages, method)
    for _ in range(epochs):
        context = encode_damage_batch(encoder, conditioning, joint_ranges, device)
        loss = _training_loss(model, states, actions, context, mask, angle)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate_surgery_gate_model(
    model: SurgeryGateModel,
    domain: DomainSpec,
    trajectories: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    device: torch.device,
    horizon: int,
) -> dict[str, float]:
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [domain.damage] * len(trajectories)
    mask, angle = _damage_tensors(damages, device)
    conditioning = _conditioning_damages(damages, model.method)
    context = encode_damage_batch(model.encoder, conditioning, joint_ranges, device)
    surgery = TopologySurgery()
    use_surgery = model.method == "topology_surgery"
    horizon = min(horizon, actions.shape[1])

    all_sq, free_sq, object_sq, violations = [], [], [], []
    dof = 5
    free_arm_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_arm_mask.sum(dim=-1).clamp_min(1.0)
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predicted = states[:, start]
        hidden = None
        for offset in range(horizon):
            action = actions[:, start + offset]
            if use_surgery:
                predicted = surgery.project_state(predicted, mask, angle)
                action = surgery.project_action(action, mask)
            output, hidden = model.world_model.step(predicted, action, context, hidden)
            raw_prediction = output["mean"]
            predicted = (
                surgery.project_state(raw_prediction, mask, angle)
                if use_surgery else raw_prediction
            )
            target = states[:, start + offset + 1]
            error_sq = (predicted - target).pow(2)
            all_sq.append(error_sq.mean(dim=-1))
            free_sq.append((error_sq[:, : 2 * dof] * free_arm_mask).sum(dim=-1) / free_count)
            object_sq.append(error_sq[:, 2 * dof :].mean(dim=-1))
            violations.append(surgery.constraint_violation(predicted, mask, angle))

    def _rmse(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values, dim=1).mean().sqrt())

    return {
        "overall_rmse": _rmse(all_sq),
        "free_arm_rmse": _rmse(free_sq),
        "object_rmse": _rmse(object_sq),
        "constraint_violation_rms": _rmse([value.pow(2) for value in violations]),
    }


def surgery_gate_parameter_count(model: SurgeryGateModel) -> int:
    return sum(parameter.numel() for parameter in model.encoder.parameters()) + sum(
        parameter.numel() for parameter in model.world_model.parameters()
    )


def train_graph_surgery_model(
    trajectories: list[SimTrajectory], *, epochs: int, device: torch.device, seed: int,
    use_topology: bool = True, hidden_dim: int = 96,
) -> TopologyGraphWorldModel:
    """Train the shared graph transition with one- and multi-step objectives."""
    torch.manual_seed(seed)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    if not use_topology:
        mask = torch.zeros_like(mask)
        angle = torch.zeros_like(angle)
    model = TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=hidden_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        hidden = None
        one_step_losses = []
        for step in range(actions.shape[1]):
            prediction, hidden = model.step(
                states[:, step], actions[:, step], mask, angle, hidden
            )
            one_step_losses.append(F.mse_loss(prediction, states[:, step + 1]))
        rollout_losses = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction = states[:, start]
            rollout_hidden = None
            for offset in range(horizon):
                prediction, rollout_hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, rollout_hidden
                )
                rollout_losses.append(F.mse_loss(prediction, states[:, start + offset + 1]))
        loss = torch.stack(one_step_losses).mean() + 0.5 * torch.stack(rollout_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate_graph_surgery_model(
    model: TopologyGraphWorldModel,
    domain: DomainSpec,
    trajectories: list[SimTrajectory],
    *, device: torch.device, horizon: int, use_topology: bool = True,
) -> dict[str, float]:
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    metric_mask, metric_angle = mask, angle
    if not use_topology:
        mask = torch.zeros_like(mask)
        angle = torch.zeros_like(angle)
    horizon = min(horizon, actions.shape[1])
    surgery = TopologySurgery()
    all_sq, free_sq, object_sq, violations = [], [], [], []
    free_arm_mask = torch.cat((1.0 - metric_mask, 1.0 - metric_mask), dim=-1)
    free_count = free_arm_mask.sum(dim=-1).clamp_min(1.0)
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction = states[:, start]
        hidden = None
        for offset in range(horizon):
            prediction, hidden = model.step(
                prediction, actions[:, start + offset], mask, angle, hidden
            )
            target = states[:, start + offset + 1]
            error_sq = (prediction - target).pow(2)
            all_sq.append(error_sq.mean(dim=-1))
            free_sq.append((error_sq[:, :10] * free_arm_mask).sum(dim=-1) / free_count)
            object_sq.append(error_sq[:, 10:].mean(dim=-1))
            violations.append(
                surgery.constraint_violation(prediction, metric_mask, metric_angle)
            )

    def _rmse(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values, dim=1).mean().sqrt())

    return {
        "overall_rmse": _rmse(all_sq), "free_arm_rmse": _rmse(free_sq),
        "object_rmse": _rmse(object_sq),
        "constraint_violation_rms": _rmse([value.pow(2) for value in violations]),
    }


def train_constraint_reaction_model(
    trajectories: list[SimTrajectory], *, base_epochs: int, reaction_epochs: int,
    device: torch.device, seed: int, hidden_dim: int = 96,
) -> ConstraintReactionWorldModel:
    base = train_graph_surgery_model(
        trajectories, epochs=base_epochs, device=device, seed=seed, use_topology=False,
        hidden_dim=hidden_dim,
    )
    model = ConstraintReactionWorldModel(base).to(device)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=2e-3)
    for _ in range(reaction_epochs):
        losses = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction = states[:, start]
            hidden = None
            for offset in range(horizon):
                prediction, hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, hidden
                )
                target = states[:, start + offset + 1]
                # Locked dimensions are analytically satisfied. Optimize only
                # the free arm and object consequences of the reaction.
                error = (prediction - target).pow(2)
                free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
                free_loss = (error[:, :10] * free_mask).sum() / free_mask.sum().clamp_min(1.0)
                object_loss = error[:, 10:].mean()
                losses.append(free_loss + object_loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    return model


def train_reduced_coordinate_graph_model(
    trajectories: list[SimTrajectory], *, epochs: int, device: torch.device,
    seed: int, hidden_dim: int = 128,
) -> ReducedCoordinateGraphWorldModel:
    """Train in the topology-dependent free-coordinate state space."""
    torch.manual_seed(seed)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    model = ReducedCoordinateGraphWorldModel(
        TopologyGraphConfig(hidden_dim=hidden_dim)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    for _ in range(epochs):
        hidden = None
        one_step_losses = []
        for step in range(actions.shape[1]):
            prediction, hidden = model.step(
                states[:, step], actions[:, step], mask, angle, hidden
            )
            error = (prediction - states[:, step + 1]).pow(2)
            free_loss = (error[:, :10] * free_mask).sum(dim=-1) / free_count
            one_step_losses.append(free_loss.mean() + error[:, 10:].mean())
        rollout_losses = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction = states[:, start]
            rollout_hidden = None
            for offset in range(horizon):
                prediction, rollout_hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, rollout_hidden
                )
                error = (prediction - states[:, start + offset + 1]).pow(2)
                free_loss = (error[:, :10] * free_mask).sum(dim=-1) / free_count
                rollout_losses.append(free_loss.mean() + error[:, 10:].mean())
        loss = torch.stack(one_step_losses).mean() + 0.5 * torch.stack(rollout_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def train_unconstrained_residual_model(
    trajectories: list[SimTrajectory], *, base_epochs: int, residual_epochs: int,
    device: torch.device, seed: int, hidden_dim: int = 96,
) -> UnconstrainedResidualGraph:
    base = train_graph_surgery_model(
        trajectories, epochs=base_epochs, device=device, seed=seed, use_topology=False,
        hidden_dim=hidden_dim,
    )
    model = UnconstrainedResidualGraph(base).to(device)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=2e-3)
    for _ in range(residual_epochs):
        losses = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction = states[:, start]
            hidden = None
            for offset in range(horizon):
                prediction, hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, hidden
                )
                losses.append(F.mse_loss(prediction, states[:, start + offset + 1]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    return model


def train_gated_reaction_model(
    trajectories: list[SimTrajectory], *, base_epochs: int, reaction_epochs: int,
    device: torch.device, seed: int, hidden_dim: int = 128,
    bottleneck_dim: int = 16, gate_logit_init: float = -4.0,
) -> GatedReactionGraph:
    base = train_graph_surgery_model(
        trajectories, epochs=base_epochs, device=device, seed=seed,
        use_topology=False, hidden_dim=hidden_dim,
    )
    model = GatedReactionGraph(
        base, bottleneck_dim=bottleneck_dim, gate_logit_init=gate_logit_init
    ).to(device)
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    mask, angle = _damage_tensors(damages, device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=2e-3)
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    for _ in range(reaction_epochs):
        losses = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction = states[:, start]
            hidden = None
            for offset in range(horizon):
                prediction, hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, hidden
                )
                error = (prediction - states[:, start + offset + 1]).pow(2)
                free_loss = (
                    (error[:, :10] * free_mask).sum()
                    / free_mask.sum().clamp_min(1.0)
                )
                losses.append(free_loss + error[:, 10:].mean())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
    return model
