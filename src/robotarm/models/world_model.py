"""Deterministic-recurrent world model (PROJECT-PLAN-V4 §4.4).

The world model predicts the next observation, reward and continuation from the
current observation, action and the damage context:

    p(o_{t+1}, r_t, continue_t | o_t, a_t, e_topology, z_residual)

Per the confirmed decision (M#1 of docs/design/models-design.md §8), G1 ships a
*deterministic* recurrent state with Gaussian output heads — deliberately
simpler than full RSSM so the mechanism can be validated end-to-end first.
Stochastic latent can be layered on later without changing the heads.

``e_topology`` and ``z_residual`` are supplied as optional conditioning; the
residual context module (M2) optimizes ``z`` against the multi-step prediction
loss exposed here. The state head predicts a delta from the current state so
autonomous rollouts remain locally anchored. Observation is modelled as a
diagonal Gaussian via separate mean / log-std heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .topology_encoder import TopologyEncoderConfig  # noqa: F401  (config shape reuse)

__all__ = ["WorldModel", "WorldModelConfig", "to_tensor_state"]


@dataclass
class WorldModelConfig:
    state_dim: int = 10  # five-joint proprioception (qpos + qvel)
    action_dim: int = 5
    latent_dim: int = 128  # recurrent hidden size
    stochastic_dim: int = 32
    # Concatenated [e_topology (64), z_residual (8)] by default.
    context_dim: int = 72
    rnn_layers: int = 1
    std_floor: float = 0.01


def to_tensor_state(x: torch.Tensor, state_dim: int) -> torch.Tensor:
    """Move a batched dict state to a flat float tensor (B, state_dim)."""
    return x.to(torch.float32).view(-1, state_dim)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        c = self.cfg

        # Input embedding: observation + action (+ context).
        in_dim = c.state_dim + c.action_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, c.latent_dim), nn.SiLU())
        self.context_proj = nn.Sequential(nn.Linear(c.context_dim, c.latent_dim), nn.SiLU())
        self.rnn = nn.GRUCell(c.latent_dim, c.latent_dim)

        self.prior_head = nn.Linear(c.latent_dim, c.stochastic_dim * 2)
        self.posterior_state = nn.Sequential(
            nn.Linear(c.state_dim, c.latent_dim),
            nn.SiLU(),
        )
        self.posterior_head = nn.Linear(
            c.latent_dim * 2, c.stochastic_dim * 2
        )
        feature_dim = c.latent_dim + c.stochastic_dim
        self.state_head = nn.Linear(feature_dim, c.state_dim * 2)
        self.reward_head = nn.Linear(feature_dim, 1)
        self.continue_head = nn.Linear(feature_dim, 1)

    @staticmethod
    def _gaussian_params(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(parameters, 2, dim=-1)
        return mean, torch.clamp(log_std, min=-5.0, max=1.0)

    def _transition(
        self,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        context: torch.Tensor | None,
        hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.encoder(torch.cat([state_t, action_t], dim=-1))
        if context is not None:
            if context.shape[-1] != self.cfg.context_dim:
                raise ValueError(
                    f"context last dimension must be {self.cfg.context_dim}, "
                    f"got {context.shape[-1]}"
                )
            x = x + self.context_proj(context)
        if hidden is None:
            hidden = torch.zeros(
                x.shape[0], self.cfg.latent_dim, device=x.device, dtype=x.dtype
            )
        return self.rnn(x, hidden)

    def _decode(
        self,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature = torch.cat([hidden, stochastic], dim=-1)
        delta, log_std = self._gaussian_params(self.state_head(feature))
        mean = state_t + delta
        reward = self.reward_head(feature).squeeze(-1)
        cont = self.continue_head(feature).squeeze(-1)
        return {
            "mean": mean,
            "log_std": log_std,
            "reward": reward,
            "continue": cont,
        }

    def step(
        self,
        state_t: torch.Tensor,  # (B, state_dim)
        action_t: torch.Tensor,  # (B, action_dim)
        context: torch.Tensor | None,  # (B, context_dim) or None
        hidden: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """One recurrent step. Returns prediction dict and new hidden."""
        hidden = self._transition(state_t, action_t, context, hidden)
        prior_mean, prior_log_std = self._gaussian_params(self.prior_head(hidden))
        prediction = self._decode(hidden, prior_mean, state_t)
        prediction["prior_mean"] = prior_mean
        prediction["prior_log_std"] = prior_log_std
        return prediction, hidden

    def observe_step(
        self,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        next_state: torch.Tensor,
        context: torch.Tensor | None,
        hidden: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Posterior training step plus a prior-only prediction for deployment."""
        hidden = self._transition(state_t, action_t, context, hidden)
        prior_mean, prior_log_std = self._gaussian_params(self.prior_head(hidden))
        next_embedding = self.posterior_state(next_state)
        posterior_mean, posterior_log_std = self._gaussian_params(
            self.posterior_head(torch.cat([hidden, next_embedding], dim=-1))
        )
        posterior_prediction = self._decode(hidden, posterior_mean, state_t)
        prior_prediction = self._decode(hidden, prior_mean, state_t)

        prior_var = torch.exp(2 * prior_log_std)
        posterior_var = torch.exp(2 * posterior_log_std)
        kl = (
            prior_log_std
            - posterior_log_std
            + (posterior_var + (posterior_mean - prior_mean).pow(2))
            / (2 * prior_var)
            - 0.5
        ).sum(dim=-1)
        posterior_prediction.update(
            {
                "prior_mean": prior_mean,
                "prior_log_std": prior_log_std,
                "posterior_mean": posterior_mean,
                "posterior_log_std": posterior_log_std,
                "kl": kl,
            }
        )
        return posterior_prediction, prior_prediction, hidden

    def nll(self, pred: dict[str, torch.Tensor], next_state: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of ``next_state`` under the predicted Gaussian."""
        std = torch.exp(pred["log_std"]).clamp_min(self.cfg.std_floor)
        diff = (next_state - pred["mean"]) / std
        return (0.5 * (diff ** 2 + 2 * pred["log_std"])).sum(dim=-1)  # (B,)

    def predict_multi_step(
        self,
        states: torch.Tensor,  # (T, state_dim)
        actions: torch.Tensor,  # (T, action_dim); aligned with actions_t -> next states
        context: torch.Tensor | None,  # (context_dim,) -> broadcast, or None
        n_future: int = 0,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run a sequence and predict future states.

        For each of the first T-1 steps we observe the true state; from step T
        onward we roll out using the model's own predicted means. Returns a
        dict with per-step NLL over the observed portion and, when
        ``n_future > 0``, a mean/reward/continue trajectory.
        """
        T = states.shape[0]
        context_b = context.unsqueeze(0) if context is not None and context.dim() == 1 else context
        nlls = []
        hidden = hidden

        def _b(x: torch.Tensor) -> torch.Tensor:
            """Add a batch dim of 1 if x is unbatched."""
            return x.unsqueeze(0) if x.dim() == 1 else x

        def _sq(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0)

        # observed portion
        for t in range(T - 1):
            pred, hidden = self.step(_b(states[t]), _b(actions[t]), context_b, hidden)
            nlls.append(_sq(self.nll(pred, _b(states[t + 1]))))
        nll = torch.stack(nlls) if nlls else torch.zeros(0, device=states.device)
        # autonomous rollout
        mean_traj = []
        last = _b(states[-1])
        a = _b(actions[min(T - 1, len(actions) - 1)])
        for _ in range(n_future):
            pred, hidden = self.step(last, a, context_b, hidden)
            last = pred["mean"].detach()
            mean_traj.append(_sq(last))
        return {
            "nll": nll,
            "future_mean": torch.stack(mean_traj) if mean_traj else torch.zeros(0, states.shape[1]),
            "hidden": hidden,
        }
