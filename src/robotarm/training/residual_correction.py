"""Training and evaluation for explicit residual dynamics correction."""
from __future__ import annotations

import torch

from robotarm.models.residual_correction import ResidualCorrection
from robotarm.models.world_model import WorldModel


def train_residual_correction(
    correction: ResidualCorrection,
    base_model: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    topology_context: torch.Tensor,
    residual_context: torch.Tensor,
    *,
    epochs: int = 40,
    lr: float = 3e-3,
    l2: float = 1e-4,
    rollout_windows: int = 4,
    rollout_weight: float = 0.1,
) -> list[float]:
    """Fit grouped corrections with normalized one- and multi-step losses."""
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=lr, weight_decay=l2)
    state_scale = states.flatten(0, 1).std(dim=0).clamp_min(1e-3)
    history = []
    generator = torch.Generator(device=states.device).manual_seed(0)
    for _ in range(epochs):
        hidden = None
        losses = []
        for step in range(actions.shape[1]):
            with torch.no_grad():
                prediction, hidden = base_model.step(
                    states[:, step], actions[:, step], topology_context, hidden
                )
            delta = correction(states[:, step], actions[:, step], residual_context)
            corrected = prediction["mean"] + delta
            error = (corrected - states[:, step + 1]) / state_scale
            losses.append(error.pow(2).mean())
        one_step_loss = torch.stack(losses).mean()

        rollout_losses = []
        for horizon in (5, 10) if rollout_weight > 0.0 else ():
            max_start = actions.shape[1] - horizon + 1
            starts = torch.randint(
                max_start,
                (min(rollout_windows, max_start),),
                generator=generator,
                device=states.device,
            )
            for start_tensor in starts:
                start = int(start_tensor)
                predicted = states[:, start]
                hidden = None
                for offset in range(horizon):
                    action = actions[:, start + offset]
                    base, hidden = base_model.step(
                        predicted, action, topology_context, hidden
                    )
                    predicted = base["mean"] + correction(
                        predicted, action, residual_context
                    )
                target = states[:, start + horizon]
                rollout_losses.append(((predicted - target) / state_scale).pow(2).mean())
        rollout_loss = (
            torch.stack(rollout_losses).mean()
            if rollout_losses
            else torch.zeros((), device=states.device)
        )
        loss = one_step_loss + rollout_weight * rollout_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(correction.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def correction_metrics(
    correction: ResidualCorrection,
    base_model: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    topology_context: torch.Tensor,
    residual_context: torch.Tensor,
    *,
    horizon: int = 10,
) -> tuple[float, float]:
    """Return teacher-forced and autonomous multi-step RMSE."""
    hidden = None
    one_step_errors = []
    for step in range(actions.shape[1]):
        prediction, hidden = base_model.step(
            states[:, step], actions[:, step], topology_context, hidden
        )
        corrected = prediction["mean"] + correction(
            states[:, step], actions[:, step], residual_context
        )
        one_step_errors.append((corrected - states[:, step + 1]).pow(2).mean(dim=-1))

    horizon = min(horizon, actions.shape[1])
    rollout_errors = []
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predicted = states[:, start]
        hidden = None
        for offset in range(horizon):
            action = actions[:, start + offset]
            base, hidden = base_model.step(predicted, action, topology_context, hidden)
            predicted = base["mean"] + correction(predicted, action, residual_context)
            rollout_errors.append(
                (predicted - states[:, start + offset + 1]).pow(2).mean(dim=-1)
            )
    one_step = torch.stack(one_step_errors, dim=1).mean().sqrt()
    multi_step = torch.stack(rollout_errors, dim=1).mean().sqrt()
    return float(one_step), float(multi_step)
