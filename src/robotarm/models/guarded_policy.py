"""Auditable carrier fallback for intervention-based control proposals."""
from __future__ import annotations

import torch


def guarded_action(
    candidate: torch.Tensor,
    carrier: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    """Publish candidate actions only for accepted samples.

    Rejected samples are bitwise selected from ``carrier``; the function does
    not interpolate, rescale, or otherwise alter the fallback action.
    """
    if candidate.shape != carrier.shape:
        raise ValueError("candidate and carrier actions must have identical shape")
    if accept.shape != candidate.shape[:-1]:
        raise ValueError("accept must contain one Boolean decision per action")
    if accept.dtype is not torch.bool:
        raise TypeError("accept must be Boolean")
    return torch.where(accept[..., None], candidate, carrier)
