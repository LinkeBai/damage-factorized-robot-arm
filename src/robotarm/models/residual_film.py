"""Low-rank residual FiLM adapters for a frozen recurrent world model."""
from __future__ import annotations

import torch
from torch import nn

from robotarm.models.world_model import WorldModel


class ResidualFiLMWorldModel(nn.Module):
    """Modulate hidden dynamics separately for arm and object predictions."""

    def __init__(
        self,
        base_model: WorldModel,
        residual_dim: int = 8,
        rank: int = 8,
        modulation_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if base_model.cfg.state_dim < 4:
            raise ValueError("state_dim must include a four-dimensional object state")
        self.base_model = base_model
        self.arm_dim = base_model.cfg.state_dim - 4
        self.modulation_scale = modulation_scale
        hidden_dim = base_model.cfg.latent_dim
        condition_dim = base_model.cfg.state_dim + base_model.cfg.action_dim + residual_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, rank),
            nn.SiLU(),
        )
        self.arm_film = nn.Linear(rank, hidden_dim * 2)
        self.object_film = nn.Linear(rank, hidden_dim * 2)
        for layer in (self.arm_film, self.object_film):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _modulate(self, hidden: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(parameters, 2, dim=-1)
        strength = self.modulation_scale
        return hidden * (1.0 + strength * torch.tanh(scale)) + strength * torch.tanh(shift)

    def _decode_path(
        self, hidden: torch.Tensor, state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prior_mean, prior_log_std = self.base_model._gaussian_params(
            self.base_model.prior_head(hidden)
        )
        prediction = self.base_model._decode(hidden, prior_mean, state)
        prediction["prior_mean"] = prior_mean
        prediction["prior_log_std"] = prior_log_std
        return prediction

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        topology_context: torch.Tensor,
        residual_context: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden = self.base_model._transition(state, action, topology_context, hidden)
        code = self.condition_encoder(
            torch.cat([state, action, residual_context], dim=-1)
        )
        arm_prediction = self._decode_path(
            self._modulate(hidden, self.arm_film(code)), state
        )
        object_prediction = self._decode_path(
            self._modulate(hidden, self.object_film(code)), state
        )
        prediction = dict(arm_prediction)
        for key in ("mean", "log_std"):
            prediction[key] = torch.cat(
                [
                    arm_prediction[key][:, : self.arm_dim],
                    object_prediction[key][:, self.arm_dim :],
                ],
                dim=-1,
            )
        return prediction, hidden
