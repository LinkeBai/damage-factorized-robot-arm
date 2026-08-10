"""Topology encoder (PROJECT-PLAN-V4 §4.2).

Encodes *discrete, diagnostic* damage topology into a fixed vector
``e_topology``. Per the plan, this must NOT be a naive per-joint lookup
embedding; instead each joint is described by stable features
``[presence, lock_angle, axis, normalized_limits, depth]``, run through a
shared MLP, and aggregated along the mechanical-chain order.

Crucially ``e_topology`` depends only on the damage description
(``joint_mask`` + ``lock_angle`` + joint attributes) and NOT on any data, so
it is available at zero-shot when an unseen topology is deployed (this is what
the topology-only baseline uses with ``z_residual = 0``).

This module is pure torch (no environment dependency) so it can be unit-tested
and imported without MuJoCo.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["TopologyEncoder", "TopologyEncoderConfig", "build_joint_features"]


@dataclass
class TopologyEncoderConfig:
    dof: int = 5
    per_joint_feat: int = 8  # presence + lock + axis(3) + limits(2) + depth
    hidden_dim: int = 64
    out_dim: int = 64
    mlp_layers: int = 2


def build_joint_features(
    joint_mask: torch.Tensor,  # (B, dof)
    lock_angle: torch.Tensor,  # (B, dof)
    axes: torch.Tensor,  # (B, dof, 3)
    limits: torch.Tensor,  # (B, dof, 2), normalized to ~[-1,1]
    depth: torch.Tensor,  # (B, dof)
) -> torch.Tensor:
    """Compose per-joint features into shape (B, dof, per_joint_feat).

    All inputs must already be batched (leading B dim); see TopologyEncoder.
    """
    presence = joint_mask.to(torch.float32).unsqueeze(-1)  # (B, dof, 1)
    lock = (lock_angle.to(torch.float32) * presence.squeeze(-1)).unsqueeze(-1)
    return torch.cat(
        [
            presence,
            lock,
            axes.to(torch.float32),
            limits.to(torch.float32),
            depth.to(torch.float32).unsqueeze(-1),  # (B, dof, 1)
        ],
        dim=-1,
    )


class TopologyEncoder(nn.Module):
    def __init__(self, cfg: TopologyEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TopologyEncoderConfig()
        c = self.cfg

        layers = [nn.Linear(c.per_joint_feat, c.hidden_dim), nn.SiLU()]
        for _ in range(c.mlp_layers - 1):
            layers += [nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU()]
        self.joint_mlp = nn.Sequential(*layers)  # shared across joints
        self.head = nn.Linear(c.hidden_dim, c.out_dim)

    def forward(
        self,
        joint_mask: torch.Tensor,  # (dof,) or (B, dof)
        lock_angle: torch.Tensor,
        axes: torch.Tensor,
        limits: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``e_topology`` of shape (B, out_dim); squeezed to (out_dim,) for a single sample.

        The head aggregates the shared-MLP joint codes along the chain order by
        mean-pooling then a final linear (G1; GNN/attention not required).
        """
        single = joint_mask.dim() == 1
        if single:
            joint_mask = joint_mask.unsqueeze(0)
            lock_angle = lock_angle.unsqueeze(0)
            axes = axes.unsqueeze(0)
            limits = limits.unsqueeze(0)
            depth = depth.unsqueeze(0)
        f = build_joint_features(joint_mask, lock_angle, axes, limits, depth)  # (B, dof, feat)
        h = self.joint_mlp(f)  # (B, dof, hidden)
        pooled = h.mean(dim=1)  # (B, hidden)
        e = self.head(pooled)  # (B, out)
        return e.squeeze(0) if single else e
