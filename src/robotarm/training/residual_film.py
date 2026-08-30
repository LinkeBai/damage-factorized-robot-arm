"""Training and evaluation for residual FiLM world-model adapters."""
from __future__ import annotations

import torch

from robotarm.models.residual_film import ResidualFiLMWorldModel


def train_residual_film(
    model: ResidualFiLMWorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    topology_context: torch.Tensor,
    residual_context: torch.Tensor,
    *,
    epochs: int = 12,
    lr: float = 3e-3,
) -> list[float]:
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scale = states.flatten(0, 1).std(dim=0).clamp_min(1e-3)
    history = []
    for _ in range(epochs):
        hidden = None
        losses = []
        for step in range(actions.shape[1]):
            prediction, hidden = model.step(
                states[:, step], actions[:, step], topology_context,
                residual_context, hidden,
            )
            error = (prediction["mean"] - states[:, step + 1]) / scale
            losses.append(error.pow(2).mean())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def residual_film_metrics(
    model: ResidualFiLMWorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    topology_context: torch.Tensor,
    residual_context: torch.Tensor,
    *,
    horizon: int = 10,
) -> tuple[float, float]:
    hidden = None
    one_step = []
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], topology_context,
            residual_context, hidden,
        )
        one_step.append((prediction["mean"] - states[:, step + 1]).pow(2).mean(dim=-1))
    rollout = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predicted = states[:, start]
        hidden = None
        for offset in range(horizon):
            prediction, hidden = model.step(
                predicted, actions[:, start + offset], topology_context,
                residual_context, hidden,
            )
            predicted = prediction["mean"]
            rollout.append(
                (predicted - states[:, start + offset + 1]).pow(2).mean(dim=-1)
            )
    return (
        float(torch.stack(one_step, dim=1).mean().sqrt()),
        float(torch.stack(rollout, dim=1).mean().sqrt()),
    )
