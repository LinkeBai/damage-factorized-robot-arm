"""Core topology-only versus DFWM mechanism benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robotarm.envs.damage import DamageConfig
from robotarm.models.history_encoder import HistoryEncoder
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
    history_topology_encoder: TopologyEncoder
    history_encoder: HistoryEncoder
    history_world_model: WorldModel
    pm_encoder: TopologyEncoder
    pm_world_model: WorldModel
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


def multi_step_rollout_rmse(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    *,
    horizon: int = 10,
) -> torch.Tensor:
    """Mean multi-step rollout RMSE using the model's own predictions.

    Unlike teacher-forcing (which feeds the true state at every step), this
    rolls the model out from each start point using its predicted mean as the
    next input. Residual-dynamics errors compound over the horizon, so this
    metric better reflects the value of an accurate residual context.
    """
    horizon = min(horizon, actions.shape[1])
    sq_errs = []
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred_state = states[:, start]
        hidden = None
        for h in range(horizon):
            prediction, hidden = wm.step(
                pred_state, actions[:, start + h], context, hidden
            )
            pred_state = prediction["mean"]
            true_state = states[:, start + h + 1]
            sq_errs.append((pred_state - true_state).pow(2).mean(dim=-1))
    rmse = torch.stack(sq_errs, dim=1).mean().sqrt()
    return rmse


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
    validation_domains: tuple[DomainSpec, ...] | None = None,
    validation_trajectories: list[SimTrajectory] | None = None,
    lr_min_ratio: float = 0.1,
    early_stop_every: int = 5,
) -> TrainedMechanismModels:
    """Train matched topology-only and DFWM predictors on identical data.

    Uses a cosine learning-rate schedule and optional validation-based early
    stopping. When ``validation_domains``/``validation_trajectories`` are given,
    the DFWM zero-shot NLL on the validation set is evaluated every
    ``early_stop_every`` epochs; the checkpoint with the lowest validation NLL
    is restored at the end, guarding against late-epoch overfitting.
    """
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
        WorldModelConfig(state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM)
    ).to(device)
    dfwm_encoder = TopologyEncoder().to(device)
    dfwm_wm = WorldModel(
        WorldModelConfig(state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM + RESIDUAL_DIM)
    ).to(device)
    train_residuals = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(train_residuals.weight)
    residual_only_wm = WorldModel(
        WorldModelConfig(state_dim=states.shape[-1], context_dim=RESIDUAL_DIM)
    ).to(device)
    residual_only_codes = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(residual_only_codes.weight)
    monolithic_encoder = TopologyEncoder().to(device)
    monolithic_wm = WorldModel(
        WorldModelConfig(state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM)
    ).to(device)
    monolithic_projection = nn.Linear(
        RESIDUAL_DIM, TOPOLOGY_DIM, bias=False
    ).to(device)
    monolithic_codes = nn.Embedding(len(train_domains), RESIDUAL_DIM).to(device)
    nn.init.zeros_(monolithic_codes.weight)

    # History encoder (amortized residual inference): outputs an 8-dim residual
    # context conditioned on the same topology encoder as DFWM, so it is a fair
    # amortized counterpart to latent optimization (§4.3B).
    history_topology_encoder = TopologyEncoder().to(device)
    history_encoder = HistoryEncoder(
        state_dim=states.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=64,
        out_dim=RESIDUAL_DIM,
    ).to(device)
    history_wm = WorldModel(
        WorldModelConfig(state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM + RESIDUAL_DIM)
    ).to(device)

    # Parameter-matched baseline: identical DFWM structure (72-dim context) but
    # the residual channel is fixed to zero during training, so the WM never
    # learns structured use of the 8 extra parameters until deployment.
    pm_encoder = TopologyEncoder().to(device)
    pm_wm = WorldModel(
        WorldModelConfig(state_dim=states.shape[-1], context_dim=TOPOLOGY_DIM + RESIDUAL_DIM)
    ).to(device)

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
        + list(history_topology_encoder.parameters())
        + list(history_encoder.parameters())
        + list(history_wm.parameters())
        + list(pm_encoder.parameters())
        + list(pm_wm.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * lr_min_ratio
    )

    # Early-stopping bookkeeping: all trainable modules, plus the DFWM
    # zero-shot validation NLL used to pick the best epoch.
    val_use = (
        validation_domains is not None and validation_trajectories is not None
    )
    if val_use:
        val_damages = [
            damage_from_name(t.domain_id.split("__", 1)[0])
            for t in validation_trajectories
        ]
        val_states, val_actions = _stack_trajectories(
            validation_trajectories, device
        )

    all_models = {
        "topology_encoder": topology_encoder,
        "topology_wm": topology_wm,
        "dfwm_encoder": dfwm_encoder,
        "dfwm_wm": dfwm_wm,
        "train_residuals": train_residuals,
        "residual_only_wm": residual_only_wm,
        "residual_only_codes": residual_only_codes,
        "monolithic_encoder": monolithic_encoder,
        "monolithic_wm": monolithic_wm,
        "monolithic_projection": monolithic_projection,
        "monolithic_codes": monolithic_codes,
        "history_topology_encoder": history_topology_encoder,
        "history_encoder": history_encoder,
        "history_wm": history_wm,
        "pm_encoder": pm_encoder,
        "pm_wm": pm_wm,
    }
    best_val_nll = float("inf")
    best_state: dict[str, dict] | None = None

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

        # History encoder: amortized residual (8-dim) conditioned on topology,
        # mirroring DFWM's [topology, z] context but with a single forward pass
        # instead of gradient optimization (§4.3B).
        history_topology_all = encode_damage_batch(
            history_topology_encoder, damages, joint_ranges, device
        )
        history_nll = torch.zeros((), device=device)
        for domain_idx in range(len(train_domains)):
            mask = domain_indices == domain_idx
            if not mask.any():
                continue
            domain_states = states[mask]
            domain_actions = actions[mask]
            domain_topology = history_topology_all[mask.nonzero()[0][0]]
            history_z = history_encoder(
                domain_states[:, :-1], domain_actions
            )
            domain_context = compose_context(
                domain_topology,
                history_z,
                context_dim=history_wm.cfg.context_dim,
            )
            context_batch = domain_context.unsqueeze(0).expand(
                domain_states.shape[0], -1
            )
            history_nll = history_nll + rssm_training_loss(
                history_wm, domain_states, domain_actions, context_batch
            )
        history_nll = history_nll / len(train_domains)

        # Parameter-matched: same 72-dim context structure as DFWM but the
        # residual channel is pinned to zero during training.
        pm_topology = encode_damage_batch(
            pm_encoder, damages, joint_ranges, device
        )
        pm_residual = torch.zeros(
            pm_topology.shape[0], RESIDUAL_DIM, device=device
        )
        pm_context = compose_context(
            pm_topology, pm_residual, context_dim=pm_wm.cfg.context_dim
        )
        pm_nll = rssm_training_loss(
            pm_wm, states, actions, pm_context
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
            + history_nll
            + pm_nll
            + latent_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "topology_nll": float(topology_nll.detach()),
                "dfwm_nll": float(dfwm_nll.detach()),
                "residual_only_nll": float(residual_only_nll.detach()),
                "monolithic_nll": float(monolithic_nll.detach()),
                "history_nll": float(history_nll.detach()),
                "pm_nll": float(pm_nll.detach()),
            }
        )

        # Early stopping: evaluate DFWM zero-shot NLL on the validation set and
        # keep the best checkpoint (guards against late-epoch overfitting).
        if val_use and (epoch + 1) % early_stop_every == 0:
            with torch.no_grad():
                val_topology = encode_damage_batch(
                    dfwm_encoder, val_damages, joint_ranges, device
                )
                val_residual = torch.zeros(
                    val_topology.shape[0], RESIDUAL_DIM, device=device
                )
                val_context = compose_context(
                    val_topology, val_residual,
                    context_dim=dfwm_wm.cfg.context_dim,
                )
                val_nll, _ = teacher_forced_metrics(
                    dfwm_wm, val_states, val_actions, val_context
                )
            val_nll_f = float(val_nll.detach())
            if val_nll_f < best_val_nll:
                best_val_nll = val_nll_f
                best_state = {
                    name: {k: v.detach().clone() for k, v in model.state_dict().items()}
                    for name, model in all_models.items()
                }

    # Restore the best-validation checkpoint if early stopping selected one.
    if best_state is not None:
        for name, model in all_models.items():
            model.load_state_dict(best_state[name])

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
        history_topology_encoder=history_topology_encoder,
        history_encoder=history_encoder,
        history_world_model=history_wm,
        pm_encoder=pm_encoder,
        pm_world_model=pm_wm,
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
        top_multi_rmse = multi_step_rollout_rmse(
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
        history_topology = encode_damage_batch(
            models.history_topology_encoder,
            [damage],
            joint_ranges,
            device,
        ).squeeze(0)
        pm_topology = encode_damage_batch(
            models.pm_encoder,
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
                "multi_step_rmse": float(top_multi_rmse),
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
            dfwm_multi_rmse = multi_step_rollout_rmse(
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
                "multi_step_rmse": float(dfwm_multi_rmse),
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

        # History encoder: amortized residual conditioned on topology. At K=0
        # the residual is zero, matching DFWM's K=0 context exactly.
        if shot == 0:
            history_z = torch.zeros(RESIDUAL_DIM, device=device)
            history_adaptation_seconds = 0.0
        else:
            started = time.perf_counter()
            with torch.no_grad():
                history_z = models.history_encoder(
                    calibration_states[:, :-1], calibration_actions
                )
            history_adaptation_seconds = time.perf_counter() - started
        history_context = compose_context(
            history_topology,
            history_z,
            context_dim=models.history_world_model.cfg.context_dim,
        )
        history_batch = history_context.unsqueeze(0).expand(
            eval_states.shape[0], -1
        )
        with torch.no_grad():
            history_nll, history_rmse = teacher_forced_metrics(
                models.history_world_model,
                eval_states,
                eval_actions,
                history_batch,
            )
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "history_encoder",
                "shots": shot,
                "eval_nll": float(history_nll),
                "eval_rmse": float(history_rmse),
                "residual_norm": float(history_z.norm()),
                "adaptation_seconds": history_adaptation_seconds,
            }
        )

        # Parameter-matched: identical DFWM structure, residual channel frozen
        # to zero during training but optimized at deployment.
        if shot == 0:
            pm_z = torch.zeros(RESIDUAL_DIM, device=device)
            pm_adaptation_seconds = 0.0
        else:
            started = time.perf_counter()
            pm_z = latent_optimize(
                models.pm_world_model,
                pm_topology,
                calibration_states,
                calibration_actions,
                LatentOptConfig(
                    d=RESIDUAL_DIM,
                    steps=latent_steps,
                    lr=0.1,
                ),
            ).z.detach()
            pm_adaptation_seconds = time.perf_counter() - started
        pm_context = compose_context(
            pm_topology,
            pm_z,
            context_dim=models.pm_world_model.cfg.context_dim,
        )
        pm_batch = pm_context.unsqueeze(0).expand(
            eval_states.shape[0], -1
        )
        with torch.no_grad():
            pm_nll, pm_rmse = teacher_forced_metrics(
                models.pm_world_model,
                eval_states,
                eval_actions,
                pm_batch,
            )
        rows.append(
            {
                "domain": domain.domain_id,
                "topology": domain.topology,
                "residual": domain.residual_name,
                "model": "parameter_matched",
                "shots": shot,
                "eval_nll": float(pm_nll),
                "eval_rmse": float(pm_rmse),
                "residual_norm": float(pm_z.norm()),
                "adaptation_seconds": pm_adaptation_seconds,
            }
        )
    return rows
