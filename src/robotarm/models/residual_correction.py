"""Explicit residual correction on top of a frozen topology dynamics model."""
from __future__ import annotations

import torch
from torch import nn


class ResidualCorrection(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        residual_dim: int = 8,
        hidden_dim: int = 128,
        delta_limit: float = 0.1,
    ) -> None:
        super().__init__()
        if state_dim < 4:
            raise ValueError("state_dim must leave room for the object-state group")
        self.arm_dim = state_dim - 4
        self.delta_limit = delta_limit
        input_dim = state_dim + action_dim + residual_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.arm_delta = nn.Linear(hidden_dim, self.arm_dim)
        self.object_delta = nn.Linear(hidden_dim, 4)
        self.gate = nn.Linear(hidden_dim, 2)
        for head in (self.arm_delta, self.object_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(0).expand(state.shape[0], -1)
        features = self.backbone(torch.cat([state, action, residual], dim=-1))
        gates = torch.sigmoid(self.gate(features))
        arm = self.delta_limit * gates[:, :1] * torch.tanh(self.arm_delta(features))
        obj = self.delta_limit * gates[:, 1:] * torch.tanh(self.object_delta(features))
        return torch.cat([arm, obj], dim=-1)
