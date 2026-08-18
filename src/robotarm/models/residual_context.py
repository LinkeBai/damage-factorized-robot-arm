"""Residual context inference — variant A: latent optimization (§4.3).

At deployment we do NOT know the true continuous residual physics of a damaged
arm (backlash, compliance, latency...). Variant A treats that residual as a
low-dimensional latent ``z_residual ∈ R^d`` and, per deployment instance:

1. initializes ``z_residual = 0``;
2. FREEZES the world model (and later the actor);
3. optimizes ONLY ``z_residual`` to minimize the multi-step prediction loss on
   the K calibrated trajectories.

This is the G1-default, lowest-risk implementation: causality is clean (only
``z`` changes), and the same frozen WM scores every morphology so deployments
are comparable. It never touches ``e_topology`` (that is fixed by diagnostic
info).

``d`` defaults to 8 (confirm-2 of the design doc); final chosen on validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .world_model import WorldModel

__all__ = [
    "LatentOptConfig",
    "ResidualContext",
    "compose_context",
    "latent_optimize",
    "latent_optimize_with_builder",
]


@dataclass
class LatentOptConfig:
    d: int = 8  # residual latent dimension
    lr: float = 1e-1
    steps: int = 50
    l2: float = 1e-3  # small prior toward zero
    max_abs: float = 5.0
    grad_clip: float | None = None
    patience: int | None = None


class ResidualContext(nn.Module):
    """A parameter ``z_residual`` that can be optimized for one deployment.

    Only ``self.z`` is a trainable parameter; everything else passed to
    ``predict`` (WM weights, topology) is detached/frozen by the caller.
    """

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.z = nn.Parameter(torch.zeros(d))
        self.initial_validation_loss: float | None = None
        self.best_validation_loss: float | None = None
        self.optimization_steps: int = 0
        self.rolled_back: bool = False

    def forward(self) -> torch.Tensor:
        return self.z

    def reset(self) -> None:
        with torch.no_grad():
            self.z.zero_()


def compose_context(
    topology: torch.Tensor,
    residual: torch.Tensor,
    *,
    context_dim: int | None = None,
) -> torch.Tensor:
    """Concatenate fixed topology and inferred residual context.

    Inputs may be unbatched or batched. A single vector is broadcast across
    the other input's batch when needed.
    """
    if topology.dim() not in (1, 2) or residual.dim() not in (1, 2):
        raise ValueError("topology and residual must be rank-1 or rank-2 tensors")
    if topology.dim() == 1 and residual.dim() == 2:
        topology = topology.unsqueeze(0).expand(residual.shape[0], -1)
    elif topology.dim() == 2 and residual.dim() == 1:
        residual = residual.unsqueeze(0).expand(topology.shape[0], -1)
    elif topology.dim() == 2 and residual.dim() == 2:
        if topology.shape[0] != residual.shape[0]:
            raise ValueError("topology and residual batch dimensions must match")

    context = torch.cat([topology, residual], dim=-1)
    if context_dim is not None and context.shape[-1] != context_dim:
        raise ValueError(
            f"topology ({topology.shape[-1]}) + residual ({residual.shape[-1]}) "
            f"must equal world-model context_dim={context_dim}"
        )
    return context


def latent_optimize(
    wm: WorldModel,
    context: torch.Tensor,  # fixed e_topology, unbatched or batched
    states: torch.Tensor,  # (K, T, state_dim) calibration trajectories
    actions: torch.Tensor,  # (K, T, action_dim)
    cfg: LatentOptConfig | None = None,
    validation_states: torch.Tensor | None = None,
    validation_actions: torch.Tensor | None = None,
) -> ResidualContext:
    """Optimize a shared residual latent over K calibration trajectories.

    ``states``/``actions`` are the K calibration trajectories, each of length T
    (so T-1 observed transitions). The WM is NOT updated — only ``z`` moves.
    Returns a ``ResidualContext`` ready to condition rollout / policy.
    """
    cfg = cfg or LatentOptConfig()
    context = context.detach()
    return latent_optimize_with_builder(
        wm,
        states,
        actions,
        lambda z: compose_context(
            context,
            z,
            context_dim=wm.cfg.context_dim,
        ),
        cfg,
        validation_states=validation_states,
        validation_actions=validation_actions,
    )


def latent_optimize_with_builder(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context_builder: Callable[[torch.Tensor], torch.Tensor],
    cfg: LatentOptConfig | None = None,
    *,
    validation_states: torch.Tensor | None = None,
    validation_actions: torch.Tensor | None = None,
) -> ResidualContext:
    """Optimize a latent through an arbitrary frozen context parameterization."""
    cfg = cfg or LatentOptConfig()
    rc = ResidualContext(cfg.d).to(device=states.device, dtype=states.dtype)
    z = rc.z

    # Freeze the world model.
    was_training = wm.training
    requires_grad = [p.requires_grad for p in wm.parameters()]
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam([z], lr=cfg.lr)

    def validation_loss() -> float | None:
        if validation_states is None or validation_actions is None:
            return None
        with torch.no_grad():
            combined = context_builder(z)
            total = 0.0
            count = 0
            for k in range(validation_states.shape[0]):
                out = wm.predict_multi_step(
                    validation_states[k], validation_actions[k], combined,
                    return_hidden=False,
                )
                total += float(out["nll"].sum())
                count += out["nll"].numel()
        return total / max(count, 1)

    initial_validation = validation_loss()
    rc.initial_validation_loss = initial_validation
    rc.best_validation_loss = initial_validation
    best_z = z.detach().clone()
    stale_steps = 0
    for step in range(cfg.steps):
        opt.zero_grad()
        total = torch.zeros((), device=z.device, dtype=torch.float32)
        n_obs = 0
        for k in range(states.shape[0]):
            combined = context_builder(z)
            if combined.shape[-1] != wm.cfg.context_dim:
                raise ValueError(
                    f"context builder returned {combined.shape[-1]} values; "
                    f"world model expects {wm.cfg.context_dim}"
                )
            out = wm.predict_multi_step(
                states[k], actions[k], combined, return_hidden=False
            )
            nll = out["nll"]  # (T-1,)
            total = total + nll.sum()
            n_obs += nll.numel()
        # prior: small pull toward the initial zero (keeps z meaningful with little data)
        prior = cfg.l2 * z.norm(p=2).pow(2)
        loss = total / max(n_obs, 1) + prior
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        opt.step()
        with torch.no_grad():
            torch.clamp_(z, -cfg.max_abs, cfg.max_abs)
        rc.optimization_steps = step + 1
        current_validation = validation_loss()
        if current_validation is not None:
            if rc.best_validation_loss is None or current_validation < rc.best_validation_loss:
                rc.best_validation_loss = current_validation
                best_z = z.detach().clone()
                stale_steps = 0
            else:
                stale_steps += 1
                if cfg.patience is not None and stale_steps >= cfg.patience:
                    break

    if initial_validation is not None:
        with torch.no_grad():
            z.copy_(best_z)
        rc.rolled_back = bool(torch.allclose(best_z, torch.zeros_like(best_z)))

    # Restore the exact state owned by the caller.
    for p, flag in zip(wm.parameters(), requires_grad):
        p.requires_grad_(flag)
    wm.train(was_training)
    return rc
