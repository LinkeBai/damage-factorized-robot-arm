"""Topology-conditioned ensembles for robust zero-shot dynamics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.g1_mechanism import (
    TOPOLOGY_DIM,
    encode_damage_batch,
    rssm_training_loss,
)
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import DomainSpec, damage_from_name


@dataclass
class TopologyMember:
    encoder: TopologyEncoder
    world_model: WorldModel


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def train_topology_ensemble(
    trajectories: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    members: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> list[TopologyMember]:
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    damages = [damage_from_name(item.domain_id.split("__", 1)[0]) for item in trajectories]
    ensemble = []
    for index in range(members):
        ensemble.append(train_topology_member(
            states, actions, damages, joint_ranges, epochs=epochs,
            device=device, seed=seed + 1009 * index, latent_dim=128,
        ))
    return ensemble


def train_topology_member(
    states: torch.Tensor,
    actions: torch.Tensor,
    damages: list,
    joint_ranges: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    latent_dim: int,
) -> TopologyMember:
    torch.manual_seed(seed)
    encoder = TopologyEncoder().to(device)
    world_model = WorldModel(WorldModelConfig(
        state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM,
        latent_dim=latent_dim,
    )).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()), lr=3e-3
    )
    for _ in range(epochs):
        context = encode_damage_batch(encoder, damages, joint_ranges, device)
        loss = rssm_training_loss(world_model, states, actions, context)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 5.0)
        optimizer.step()
    return TopologyMember(encoder=encoder, world_model=world_model)


def member_parameter_count(member: TopologyMember) -> int:
    return sum(p.numel() for p in member.encoder.parameters()) + sum(
        p.numel() for p in member.world_model.parameters()
    )


def matched_latent_dim(state_dim: int, target_parameters: int) -> int:
    candidates = range(128, 385, 8)
    counts = {}
    for latent_dim in candidates:
        probe = TopologyMember(
            TopologyEncoder(),
            WorldModel(WorldModelConfig(
                state_dim=state_dim, context_dim=TOPOLOGY_DIM,
                latent_dim=latent_dim,
            )),
        )
        counts[latent_dim] = member_parameter_count(probe)
    return min(candidates, key=lambda value: abs(counts[value] - target_parameters))


@torch.no_grad()
def evaluate_topology_ensemble(
    ensemble: list[TopologyMember],
    domain: DomainSpec,
    trajectories: list[SimTrajectory],
    joint_ranges: np.ndarray,
    *,
    device: torch.device,
    horizon: int = 10,
) -> dict[str, float]:
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    contexts = [
        encode_damage_batch(member.encoder, [domain.damage] * len(trajectories), joint_ranges, device)
        for member in ensemble
    ]
    member_errors = [[] for _ in ensemble]
    ensemble_errors = []
    uncertainties = []
    aleatoric_uncertainties = []
    total_uncertainties = []
    realized_errors = []
    horizon = min(horizon, actions.shape[1])
    uncertainty_by_depth = [[] for _ in range(horizon)]
    aleatoric_by_depth = [[] for _ in range(horizon)]
    total_by_depth = [[] for _ in range(horizon)]
    error_by_depth = [[] for _ in range(horizon)]
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predictions = [states[:, start].clone() for _ in ensemble]
        hidden = [None for _ in ensemble]
        for offset in range(horizon):
            means = []
            variances = []
            for index, member in enumerate(ensemble):
                output, hidden[index] = member.world_model.step(
                    predictions[index], actions[:, start + offset], contexts[index], hidden[index]
                )
                predictions[index] = output["mean"]
                means.append(output["mean"])
                variances.append(torch.exp(2.0 * output["log_std"]))
            stacked = torch.stack(means)
            mean_prediction = stacked.mean(dim=0)
            target = states[:, start + offset + 1]
            for index, prediction in enumerate(means):
                member_errors[index].append((prediction - target).pow(2).mean(dim=-1))
            error = (mean_prediction - target).pow(2).mean(dim=-1).sqrt()
            uncertainty = stacked.var(dim=0, unbiased=False).mean(dim=-1).sqrt()
            aleatoric = torch.stack(variances).mean(dim=(0, 2)).sqrt()
            total_uncertainty = (uncertainty.pow(2) + aleatoric.pow(2)).sqrt()
            ensemble_errors.append(error.pow(2))
            uncertainties.append(uncertainty)
            aleatoric_uncertainties.append(aleatoric)
            total_uncertainties.append(total_uncertainty)
            realized_errors.append(error)
            uncertainty_by_depth[offset].append(uncertainty)
            aleatoric_by_depth[offset].append(aleatoric)
            total_by_depth[offset].append(total_uncertainty)
            error_by_depth[offset].append(error)
    uncertainty = torch.cat(uncertainties).cpu().numpy()
    error = torch.cat(realized_errors).cpu().numpy()
    aleatoric_uncertainty = torch.cat(aleatoric_uncertainties).cpu().numpy()
    total_uncertainty = torch.cat(total_uncertainties).cpu().numpy()
    correlation = _spearman(uncertainty, error)
    depth_correlations = [
        _spearman(
            torch.cat(uncertainty_by_depth[depth]).cpu().numpy(),
            torch.cat(error_by_depth[depth]).cpu().numpy(),
        )
        for depth in range(horizon)
    ]
    aleatoric_depth_correlations = [
        _spearman(
            torch.cat(aleatoric_by_depth[depth]).cpu().numpy(),
            torch.cat(error_by_depth[depth]).cpu().numpy(),
        )
        for depth in range(horizon)
    ]
    total_depth_correlations = [
        _spearman(
            torch.cat(total_by_depth[depth]).cpu().numpy(),
            torch.cat(error_by_depth[depth]).cpu().numpy(),
        )
        for depth in range(horizon)
    ]
    member_rmse = [float(torch.stack(values, dim=1).mean().sqrt()) for values in member_errors]
    return {
        "ensemble_rmse": float(torch.stack(ensemble_errors, dim=1).mean().sqrt()),
        "mean_member_rmse": float(np.mean(member_rmse)),
        "best_member_rmse": float(np.min(member_rmse)),
        "uncertainty_error_spearman": correlation,
        "depth_stratified_spearman": float(np.mean(depth_correlations)),
        "aleatoric_depth_spearman": float(np.mean(aleatoric_depth_correlations)),
        "total_depth_spearman": float(np.mean(total_depth_correlations)),
        "aleatoric_error_spearman": _spearman(aleatoric_uncertainty, error),
        "total_error_spearman": _spearman(total_uncertainty, error),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }
