"""Conservative deployment-time optimization of a residual context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class SafeAdaptConfig:
    latent_dim: int = 8
    steps: int = 12
    initial_step_size: float = 0.2
    backtracking_factor: float = 0.5
    backtracking_steps: int = 8
    trust_radius: float = 0.5
    l2: float = 1e-3
    validation_tolerance: float = 0.01
    minimum_validation_improvement: float = 1e-4
    gradient_epsilon: float = 1e-8


@dataclass
class SafeAdaptResult:
    z: torch.Tensor
    initial_fit_loss: float
    initial_validation_loss: float
    best_validation_loss: float
    accepted_steps: int
    attempted_steps: int
    rolled_back: bool
    history: list[dict[str, float | int | bool]]


def _project_ball(value: torch.Tensor, radius: float) -> torch.Tensor:
    norm = value.norm()
    if float(norm) <= radius:
        return value
    return value * (radius / norm.clamp_min(1e-12))


def safe_adapt_residual(
    fit_loss: Callable[[torch.Tensor], torch.Tensor],
    validation_loss: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    config: SafeAdaptConfig | None = None,
) -> SafeAdaptResult:
    """Optimize ``z`` only when fit and held-out calibration losses agree.

    Gradient normalization removes deployment-dependent loss scale. Every
    update is backtracked, constrained to a ball around the K=0 context, and
    rejected unless it improves fit without materially degrading validation.
    The returned context is zero when validation cannot beat the K=0 model.
    """
    cfg = config or SafeAdaptConfig()
    z = torch.zeros(cfg.latent_dim, device=device, dtype=dtype)
    with torch.no_grad():
        initial_fit = float(fit_loss(z))
        initial_validation = float(validation_loss(z))
    best_z = z.clone()
    best_validation = initial_validation
    accepted = 0
    history: list[dict[str, float | int | bool]] = []
    for step in range(cfg.steps):
        current = z.detach().requires_grad_(True)
        objective = fit_loss(current) + cfg.l2 * current.pow(2).mean()
        gradient, = torch.autograd.grad(objective, current)
        gradient_norm = gradient.norm()
        if not torch.isfinite(gradient_norm) or float(gradient_norm) < cfg.gradient_epsilon:
            history.append({"step": step, "accepted": False,
                            "gradient_norm": float(gradient_norm)})
            break
        direction = gradient / gradient_norm.clamp_min(cfg.gradient_epsilon)
        with torch.no_grad():
            current_fit = float(fit_loss(z))
            current_validation = float(validation_loss(z))
        chosen = None
        for backtrack in range(cfg.backtracking_steps):
            step_size = cfg.initial_step_size * cfg.backtracking_factor ** backtrack
            candidate = _project_ball(z - step_size * direction.detach(), cfg.trust_radius)
            with torch.no_grad():
                candidate_fit = float(fit_loss(candidate))
                candidate_validation = float(validation_loss(candidate))
            fit_ok = candidate_fit < current_fit
            validation_limit = min(current_validation, best_validation) * (
                1.0 + cfg.validation_tolerance)
            validation_ok = candidate_validation <= validation_limit
            if (fit_ok and validation_ok and torch.isfinite(
                    torch.tensor(candidate_fit + candidate_validation))):
                chosen = (candidate, candidate_fit, candidate_validation,
                          step_size, backtrack)
                break
        if chosen is None:
            history.append({"step": step, "accepted": False,
                            "gradient_norm": float(gradient_norm)})
            break
        z, fit_value, validation_value, step_size, backtrack = chosen
        accepted += 1
        if validation_value < best_validation:
            best_validation = validation_value
            best_z = z.clone()
        history.append({"step": step, "accepted": True,
                        "gradient_norm": float(gradient_norm),
                        "step_size": step_size, "backtracks": backtrack,
                        "fit_loss": fit_value,
                        "validation_loss": validation_value,
                        "z_norm": float(z.norm())})
    relative_gain = ((initial_validation - best_validation)
                     / max(abs(initial_validation), 1e-12))
    rolled_back = relative_gain < cfg.minimum_validation_improvement
    final_z = torch.zeros_like(best_z) if rolled_back else best_z
    return SafeAdaptResult(
        z=final_z, initial_fit_loss=initial_fit,
        initial_validation_loss=initial_validation,
        best_validation_loss=best_validation, accepted_steps=accepted,
        attempted_steps=len(history), rolled_back=rolled_back, history=history,
    )
