"""Core topology-only versus DFWM mechanism benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robotarm.envs.damage import DamageConfig
from robotarm.models.residual_context import (
    LatentOptConfig,
    compose_context,
    latent_optimize,
    latent_optimize_with_builder,
)
from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import DomainSpec, damage_from_name

TOPOLOGY_DIM = 64
RESIDUAL_DIM = 8


@dataclass
class TrainedMechanismModels:
    topology_encoder: TopologyEncoder
    topology_world_model: WorldModel
    dfwm_encoder: TopologyEncoder
    dfwm_world_model: WorldModel
    train_residuals: nn.Embedding
    residual_only_world_model: WorldModel
    residual_only_codes: nn.Embedding
    monolithic_encoder: TopologyEncoder
    monolithic_world_model: WorldModel
    monolithic_projection: nn.Linear
    monolithic_codes: nn.Embedding
    domain_to_index: dict[str, int]
    history: list[dict[str, float]]


def encode_damage_batch(
    encoder: TopologyEncoder,
    damages: list[DamageConfig],
    joint_ranges: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    batch = len(damages)
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
    axes = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(batch, -1, -1)
    limits = (
        torch.as_tensor(joint_ranges / np.pi, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(batch, -1, -1)
    )
    depth = torch.linspace(0.0, 1.0, 5, device=device).unsqueeze(0).expand(batch, -1)
    return encoder(masks, angles, axes, limits, depth)


def _stack_trajectories(
    trajectories: list[SimTrajectory], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.stack([trajectory.states for trajectory in trajectories]).to(device)
    actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(device)
    return states, actions


def teacher_forced_metrics(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean Gaussian NLL and one-step state RMSE for a batch."""
    hidden = None
    nlls = []
    squared_errors = []
    for step in range(actions.shape[1]):
        prediction, hidden = wm.step(
            states[:, step], actions[:, step], context, hidden
        )
        next_state = states[:, step + 1]
        nlls.append(wm.nll(prediction, next_state))
        squared_errors.append((prediction["mean"] - next_state).pow(2).mean(dim=-1))
    nll = torch.stack(nlls, dim=1).mean()
    rmse = torch.stack(squared_errors, dim=1).mean().sqrt()
    return nll, rmse


def rssm_training_loss(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    *,
    kl_weight: float = 0.01,
) -> torch.Tensor:
    """Train posterior reconstruction while keeping the deployment prior useful."""
    hidden = None
    losses = []
    for step in range(actions.shape[1]):
        posterior, prior, hidden = wm.observe_step(
            states[:, step],
            actions[:, step],
            states[:, step + 1],
            context,
            hidden,
        )
        posterior_nll = wm.nll(posterior, states[:, step + 1])
        prior_nll = wm.nll(prior, states[:, step + 1])
        losses.append(
            posterior_nll + 0.5 * prior_nll + kl_weight * posterior["kl"]
        )
    one_step = torch.stack(losses, dim=1).mean()
    rollout_losses = []
    horizon = min(5, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predicted = states[:, start]
        rollout_hidden = None
        for offset in range(horizon):
            prediction, rollout_hidden = wm.step(
                predicted,
                actions[:, start + offset],
                context,
                rollout_hidden,
            )
            predicted = prediction["mean"]
            rollout_losses.append(
                F.mse_loss(predicted, states[:, start + offset + 1])
            )
    multi_step = torch.stack(rollout_losses).mean()
    return one_step + 0.5 * multi_step


def train_mechanism_models(
    train_domains: tuple[DomainSpec, ...],
    trajectories: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 3e-3,
) -> TrainedMechanismModels:
    """Train matched topology-only and DFWM predictors on identical data."""
    domain_to_index = {
        domain.domain_id: index for index, domain in enumerate(train_domains)
    }
    damages = [
        damage_from_name(trajectory.domain_id.split("__", 1)[0])
        for trajectory in trajectories
    ]
    domain_indices = torch.tensor(
        [domain_to_index[trajectory.domain_id] for trajectory in trajectories],
        dtype=torch.long,
        device=device,
    )
    states, actions = _stack_trajectories(trajectories, device)

    topology_encoder = TopologyEncoder().to(device)
    topology_wm = WorldModel(
        WorldModelConfig(context_dim=TOPOLOGY_DIM)
    ).to(device)
    dfwm_encoder = TopologyEncoder().to(device)
    dfwm_wm = WorldModel(
        WorldModelConfig(context_dim=TOPOLOGY_DIM + RESIDUAL_DIM)
    ).to(device)
    train_residuals = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(train_residuals.weight)
    residual_only_wm = WorldModel(
        WorldModelConfig(context_dim=RESIDUAL_DIM)
    ).to(device)
    residual_only_codes = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(residual_only_codes.weight)
    monolithic_encoder = TopologyEncoder().to(device)
    monolithic_wm = WorldModel(
        WorldModelConfig(context_dim=TOPOLOGY_DIM)
    ).to(device)
    monolithic_projection = nn.Linear(
        RESIDUAL_DIM, TOPOLOGY_DIM, bias=False
    ).to(device)
    monolithic_codes = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(monolithic_codes.weight)

    parameters = (
        list(topology_encoder.parameters())
        + list(topology_wm.parameters())
        + list(dfwm_encoder.parameters())
        + list(dfwm_wm.parameters())
        + list(train_residuals.parameters())
        + list(residual_only_wm.parameters())
        + list(residual_only_codes.parameters())
        + list(monolithic_encoder.parameters())
        + list(monolithic_wm.parameters())
        + list(monolithic_projection.parameters())
        + list(monolithic_codes.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        topology_context = encode_damage_batch(
            topology_encoder, damages, joint_ranges, device
        )
        topology_nll = rssm_training_loss(
            topology_wm, states, actions, topology_context
        )

        dfwm_topology = encode_damage_batch(
            dfwm_encoder, damages, joint_ranges, device
        )
        residual = train_residuals(domain_indices)
        dfwm_context = compose_context(
            dfwm_topology,
            residual,
            context_dim=dfwm_wm.cfg.context_dim,
        )
        dfwm_nll = rssm_training_loss(
            dfwm_wm, states, actions, dfwm_context
        )
        residual_only = residual_only_codes(domain_indices)
        residual_only_nll = rssm_training_loss(
            residual_only_wm,
            states,
            actions,
            residual_only,
        )
        monolithic_topology = encode_damage_batch(
            monolithic_encoder, damages, joint_ranges, device
        )
        monolithic_residual = monolithic_codes(domain_indices)
        monolithic_context = (
            monolithic_topology
            + monolithic_projection(monolithic_residual)
        )
        monolithic_nll = rssm_training_loss(
            monolithic_wm,
            states,
            actions,
            monolithic_context,
        )
        latent_prior = 1e-3 * (
            residual.pow(2).mean()
            + residual_only.pow(2).mean()
            + monolithic_residual.pow(2).mean()
        )
        loss = (
            topology_nll
            + dfwm_nll
            + residual_only_nll
            + monolithic_nll
            + latent_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch),
                "topology_nll": float(topology_nll.detach()),
                "dfwm_nll": float(dfwm_nll.detach()),
                "residual_only_nll": float(residual_only_nll.detach()),
                "monolithic_nll": float(monolithic_nll.detach()),
            }
        )

    return TrainedMechanismModels(
        topology_encoder=topology_encoder,
        topology_world_model=topology_wm,
        dfwm_encoder=dfwm_encoder,
        dfwm_world_model=dfwm_wm,
        train_residuals=train_residuals,
        residual_only_world_model=residual_only_wm,
        residual_only_codes=residual_only_codes,
        monolithic_encoder=monolithic_encoder,
        monolithic_world_model=monolithic_wm,
        monolithic_projection=monolithic_projection,
        monolithic_codes=monolithic_codes,
        domain_to_index=domain_to_index,
        history=history,
    )


def evaluate_test_domain(
    models: TrainedMechanismModels,
    domain: DomainSpec,
    calibration: list[SimTrajectory],
    evaluation: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    shots: tuple[int, ...],
    latent_steps: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    """Evaluate on trajectories disjoint from latent calibration data."""
    eval_states, eval_actions = _stack_trajectories(evaluation, device)
    damage = domain.damage
    with torch.no_grad():
        topology_context = encode_damage_batch(
            models.topology_encoder,
            [damage],
            joint_ranges,
            device,
        ).squeeze(0)
        topology_batch = topology_context.unsqueeze(0).expand(eval_states.shape[0], -1)
        top_nll, top_rmse = teacher_forced_metrics(
            models.topology_world_model,
            eval_states,
            eval_actions,
            topology_batch,
        )

        dfwm_topology = encode_damage_batch(
            models.dfwm_encoder,
            [damage],
            joint_ranges,
            device,
        ).squeeze(0)
        monolithic_topology = encode_damage_batch(
            models.monolithic_encoder,
            [damage],
            joint_ranges,
            device,
        ).squeeze(0)

    rows: list[dict[str, float | int | str]] = []
    for shot in shots:
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "topology_only",
                "shots": shot,
                "eval_nll": float(top_nll),
                "eval_rmse": float(top_rmse),
                "residual_norm": 0.0,
                "adaptation_seconds": 0.0,
            }
        )
        if shot > 0:
            selected = calibration[:shot]
            calibration_states, calibration_actions = _stack_trajectories(
                selected, device
            )
        if shot == 0:
            z = torch.zeros(RESIDUAL_DIM, device=device)
            dfwm_adaptation_seconds = 0.0
        else:
            started = time.perf_counter()
            inferred = latent_optimize(
                models.dfwm_world_model,
                dfwm_topology,
                calibration_states,
                calibration_actions,
                LatentOptConfig(
                    d=RESIDUAL_DIM,
                    steps=latent_steps,
                    lr=0.1,
                ),
            )
            z = inferred.z.detach()
            dfwm_adaptation_seconds = time.perf_counter() - started
        context = compose_context(
            dfwm_topology,
            z,
            context_dim=models.dfwm_world_model.cfg.context_dim,
        )
        context_batch = context.unsqueeze(0).expand(eval_states.shape[0], -1)
        with torch.no_grad():
            dfwm_nll, dfwm_rmse = teacher_forced_metrics(
                models.dfwm_world_model,
                eval_states,
                eval_actions,
                context_batch,
            )
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "dfwm",
                "shots": shot,
                "eval_nll": float(dfwm_nll),
                "eval_rmse": float(dfwm_rmse),
                "residual_norm": float(z.norm()),
                "adaptation_seconds": dfwm_adaptation_seconds,
            }
        )

        if shot == 0:
            residual_only_z = torch.zeros(RESIDUAL_DIM, device=device)
            residual_adaptation_seconds = 0.0
        else:
            started = time.perf_counter()
            residual_only_z = latent_optimize_with_builder(
                models.residual_only_world_model,
                calibration_states,
                calibration_actions,
                lambda value: value,
                LatentOptConfig(
                    d=RESIDUAL_DIM,
                    steps=latent_steps,
                    lr=0.1,
                ),
            ).z.detach()
            residual_adaptation_seconds = time.perf_counter() - started
        residual_context = residual_only_z.unsqueeze(0).expand(
            eval_states.shape[0], -1
        )
        with torch.no_grad():
            residual_nll, residual_rmse = teacher_forced_metrics(
                models.residual_only_world_model,
                eval_states,
                eval_actions,
                residual_context,
            )
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "residual_only",
                "shots": shot,
                "eval_nll": float(residual_nll),
                "eval_rmse": float(residual_rmse),
                "residual_norm": float(residual_only_z.norm()),
                "adaptation_seconds": residual_adaptation_seconds,
            }
        )

        projection_weight = models.monolithic_projection.weight.detach()

        def monolithic_builder(value: torch.Tensor) -> torch.Tensor:
            return monolithic_topology + F.linear(value, projection_weight)

        if shot == 0:
            monolithic_z = torch.zeros(RESIDUAL_DIM, device=device)
            monolithic_adaptation_seconds = 0.0
        else:
            started = time.perf_counter()
            monolithic_z = latent_optimize_with_builder(
                models.monolithic_world_model,
                calibration_states,
                calibration_actions,
                monolithic_builder,
                LatentOptConfig(
                    d=RESIDUAL_DIM,
                    steps=latent_steps,
                    lr=0.1,
                ),
            ).z.detach()
            monolithic_adaptation_seconds = time.perf_counter() - started
        monolithic_context = monolithic_builder(monolithic_z)
        monolithic_batch = monolithic_context.unsqueeze(0).expand(
            eval_states.shape[0], -1
        )
        with torch.no_grad():
            monolithic_nll, monolithic_rmse = teacher_forced_metrics(
                models.monolithic_world_model,
                eval_states,
                eval_actions,
                monolithic_batch,
            )
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "monolithic_matched",
                "shots": shot,
                "eval_nll": float(monolithic_nll),
                "eval_rmse": float(monolithic_rmse),
                "residual_norm": float(monolithic_z.norm()),
                "adaptation_seconds": monolithic_adaptation_seconds,
            }
        )
    return rows
