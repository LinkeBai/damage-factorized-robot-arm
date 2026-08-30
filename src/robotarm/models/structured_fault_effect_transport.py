"""Structure-preserving few-shot transport of task effects after a hard lock.

The module does not learn another full transition model.  It reuses a world
model's local action-to-task-effect Jacobian, removes the locked action column,
updates only the remaining response map from a few observed fault secants, and
solves a regularized inverse problem that preserves the nominal task effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SFETConfig:
    ridge: float = 1e-3
    secant_epsilon: float = 1e-8
    action_limit: float = 1.0


class StructuredFaultEffectTransport:
    """Few-shot Broyden adaptation plus hard-constrained action repair."""

    def __init__(
        self,
        effect_jacobian: npt.ArrayLike,
        *,
        locked: tuple[int, ...],
        config: SFETConfig | None = None,
    ) -> None:
        jacobian = np.asarray(effect_jacobian, dtype=np.float64)
        if jacobian.ndim != 2:
            raise ValueError("effect_jacobian must be a matrix")
        self.config = config or SFETConfig()
        self.effect_dim, self.action_dim = jacobian.shape
        if any(index < 0 or index >= self.action_dim for index in locked):
            raise ValueError("locked action index is out of range")
        self.locked = tuple(sorted(set(int(index) for index in locked)))
        self.free_mask = np.ones(self.action_dim, dtype=np.float64)
        self.free_mask[list(self.locked)] = 0.0
        self.jacobian = jacobian.copy()
        self.jacobian[:, list(self.locked)] = 0.0

    def update(self, action_delta: npt.ArrayLike, effect_delta: npt.ArrayLike) -> float:
        """Apply a masked good-Broyden secant update and return residual norm."""
        action = np.asarray(action_delta, dtype=np.float64).reshape(self.action_dim)
        effect = np.asarray(effect_delta, dtype=np.float64).reshape(self.effect_dim)
        free_action = action * self.free_mask
        residual = effect - self.jacobian @ free_action
        denominator = float(free_action @ free_action) + self.config.secant_epsilon
        if denominator > self.config.secant_epsilon:
            self.jacobian += np.outer(residual, free_action) / denominator
            self.jacobian[:, list(self.locked)] = 0.0
        return float(np.linalg.norm(residual))

    def repair(
        self,
        nominal_action: npt.ArrayLike,
        desired_effect: npt.ArrayLike,
        masked_action_effect: npt.ArrayLike,
    ) -> Array:
        """Return the minimum-change free-joint action matching task effect."""
        nominal = np.asarray(nominal_action, dtype=np.float64).reshape(self.action_dim)
        desired = np.asarray(desired_effect, dtype=np.float64).reshape(self.effect_dim)
        current = np.asarray(masked_action_effect, dtype=np.float64).reshape(self.effect_dim)
        masked = nominal * self.free_mask
        response = self.jacobian * self.free_mask[None, :]
        system = response @ response.T + self.config.ridge * np.eye(self.effect_dim)
        correction = response.T @ np.linalg.solve(system, desired - current)
        repaired = np.clip(
            masked + correction,
            -self.config.action_limit,
            self.config.action_limit,
        )
        repaired[list(self.locked)] = 0.0
        return repaired

    def predicted_effect_change(self, action_delta: npt.ArrayLike) -> Array:
        action = np.asarray(action_delta, dtype=np.float64).reshape(self.action_dim)
        return self.jacobian @ (action * self.free_mask)


def ipwm_effect_jacobian(
    model: torch.nn.Module,
    state: torch.Tensor,
    action: torch.Tensor,
    lock_mask: torch.Tensor,
    lock_angle: torch.Tensor,
    *,
    effect_indices: tuple[int, ...] = (10, 11),
) -> Array:
    """Differentiate one IPWM step's task effect with respect to action."""
    if state.ndim != 1 or action.ndim != 1:
        raise ValueError("state and action must be unbatched vectors")
    differentiable_action = action.detach().clone().requires_grad_(True)

    def effect(candidate: torch.Tensor) -> torch.Tensor:
        next_state, _ = model.step(
            state.unsqueeze(0),
            candidate.unsqueeze(0),
            lock_mask.unsqueeze(0),
            lock_angle.unsqueeze(0),
            None,
        )
        return next_state[0, list(effect_indices)] - state[list(effect_indices)]

    jacobian = torch.autograd.functional.jacobian(effect, differentiable_action)
    return jacobian.detach().cpu().double().numpy()
