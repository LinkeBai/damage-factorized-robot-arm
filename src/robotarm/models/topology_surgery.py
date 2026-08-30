"""Hard topology surgery for known joint-locking failures.

The operator edits the transition inputs and outputs instead of presenting the
failure as a soft embedding that a world model can ignore. Push observations
are laid out as ``[q(5), qvel(5), object_state(4)]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopologySurgerySpec:
    dof: int = 5


class TopologySurgery:
    def __init__(self, spec: TopologySurgerySpec | None = None) -> None:
        self.spec = spec or TopologySurgerySpec()

    def _validate(
        self, state: torch.Tensor, mask: torch.Tensor, lock_angle: torch.Tensor
    ) -> None:
        dof = self.spec.dof
        if state.shape[-1] < 2 * dof:
            raise ValueError(f"state must contain q and qvel for {dof} joints")
        if mask.shape[-1] != dof or lock_angle.shape[-1] != dof:
            raise ValueError(f"mask and lock_angle must end in dimension {dof}")

    def project_state(
        self, state: torch.Tensor, mask: torch.Tensor, lock_angle: torch.Tensor
    ) -> torch.Tensor:
        """Enforce locked position and zero velocity without touching objects."""
        self._validate(state, mask, lock_angle)
        dof = self.spec.dof
        mask = mask.to(device=state.device, dtype=state.dtype)
        lock_angle = lock_angle.to(device=state.device, dtype=state.dtype)
        q = state[..., :dof] * (1.0 - mask) + lock_angle * mask
        qvel = state[..., dof : 2 * dof] * (1.0 - mask)
        return torch.cat((q, qvel, state[..., 2 * dof :]), dim=-1)

    def project_action(self, action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Remove commands on locked actuators while retaining other commands."""
        dof = self.spec.dof
        if action.shape[-1] != dof or mask.shape[-1] != dof:
            raise ValueError(f"action and mask must end in dimension {dof}")
        mask = mask.to(device=action.device, dtype=action.dtype)
        return action * (1.0 - mask)

    def constraint_violation(
        self, state: torch.Tensor, mask: torch.Tensor, lock_angle: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample RMS violation over locked position and velocity entries."""
        self._validate(state, mask, lock_angle)
        dof = self.spec.dof
        mask = mask.to(device=state.device, dtype=state.dtype)
        lock_angle = lock_angle.to(device=state.device, dtype=state.dtype)
        count = (2.0 * mask.sum(dim=-1)).clamp_min(1.0)
        pos = ((state[..., :dof] - lock_angle) * mask).pow(2).sum(dim=-1)
        vel = (state[..., dof : 2 * dof] * mask).pow(2).sum(dim=-1)
        return ((pos + vel) / count).sqrt()
