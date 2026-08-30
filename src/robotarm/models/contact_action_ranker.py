"""Contact-conditioned action-sequence ranker for carrier-policy residual planning."""
from __future__ import annotations

import torch
from torch import nn


class ContactActionRanker(nn.Module):
    """Assign a lower score to action sequences with lower realized cost."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def pairwise_ranking_loss(scores: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Logistic ordering loss; lower score must correspond to lower cost."""
    count = scores.shape[0]
    first, second = torch.triu_indices(count, count, offset=1, device=scores.device)
    cost_delta = costs[second] - costs[first]
    informative = cost_delta.abs() > 1e-8
    if not informative.any():
        return scores.sum() * 0.0
    desired = torch.sign(cost_delta[informative])
    score_delta = scores[second[informative]] - scores[first[informative]]
    return torch.nn.functional.softplus(-desired * score_delta).mean()
