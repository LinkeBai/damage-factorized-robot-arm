"""Variable-budget amortized inference of physically grounded residual context."""
from __future__ import annotations

import torch
from torch import nn


class PhysicalContextEncoder(nn.Module):
    """Permutation-invariant transition encoder with a fixed physical output basis."""

    def __init__(self, state_dim=14, action_dim=5, topology_dim=5,
                 hidden_dim=96, context_dim=8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.topology_dim = topology_dim
        feature_dim = 2 * state_dim + action_dim + topology_dim
        self.transition_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.context_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim))

    def forward(self, states, actions, damage_mask):
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("states must contain exactly one more step than actions")
        if damage_mask.dim() == 1:
            damage_mask = damage_mask.unsqueeze(0).expand(states.shape[0], -1)
        delta = states[:, 1:] - states[:, :-1]
        mask = damage_mask[:, None, :].expand(-1, actions.shape[1], -1)
        features = torch.cat((states[:, :-1], actions, delta, mask), dim=-1)
        encoded = self.transition_encoder(features)
        pooled = torch.cat((encoded.mean(1), encoded.var(1, unbiased=False)), dim=-1)
        return self.context_head(pooled)


class UncertainPhysicalContextEncoder(PhysicalContextEncoder):
    """Physical context posterior with diagonal, observation-dependent variance."""

    def __init__(self, state_dim=14, action_dim=5, topology_dim=5,
                 hidden_dim=96, context_dim=8):
        super().__init__(state_dim, action_dim, topology_dim,
                         hidden_dim, context_dim)
        self.log_variance_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, context_dim))

    def pooled_features(self, states, actions, damage_mask):
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("states must contain exactly one more step than actions")
        if damage_mask.dim() == 1:
            damage_mask = damage_mask.unsqueeze(0).expand(states.shape[0], -1)
        delta = states[:, 1:] - states[:, :-1]
        mask = damage_mask[:, None, :].expand(-1, actions.shape[1], -1)
        features = torch.cat((states[:, :-1], actions, delta, mask), dim=-1)
        encoded = self.transition_encoder(features)
        return torch.cat((encoded.mean(1), encoded.var(1, unbiased=False)), dim=-1)

    def forward(self, states, actions, damage_mask, return_uncertainty=False):
        pooled = self.pooled_features(states, actions, damage_mask)
        mean = self.context_head(pooled)
        log_variance = self.log_variance_head(pooled).clamp(-7.0, 3.0)
        return (mean, log_variance) if return_uncertainty else mean
