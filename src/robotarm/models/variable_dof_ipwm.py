"""Variable-DoF structural interface for the unchanged IPWM core mechanism.

This module is an interface feasibility gate, not a new performance component.
It expresses serial-chain joint states as valid nodes, applies the same hard
lock projection for any chain length, and uses shared message/update weights.
Padding is batching-only: invalid nodes cannot send or receive messages and
must not change predictions on valid nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SerialChainSpec:
    name: str
    joint_names: tuple[str, ...]
    axes: np.ndarray
    origins: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @classmethod
    def from_mjcf(
        cls, path: str | Path, joint_names: tuple[str, ...], *, name: str
    ) -> "SerialChainSpec":
        model = mujoco.MjModel.from_xml_path(str(Path(path)))
        ids = np.array([model.joint(joint).id for joint in joint_names], dtype=int)
        axes = np.asarray(model.jnt_axis[ids], dtype=np.float32).copy()
        origins = np.asarray(model.jnt_pos[ids], dtype=np.float32).copy()
        if axes.shape != (len(joint_names), 3) or origins.shape != axes.shape:
            raise ValueError("invalid serial-chain joint geometry")
        return cls(name=name, joint_names=joint_names, axes=axes, origins=origins)


def pad_serial_batch(
    values: list[torch.Tensor], *, width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad [Ni, D] tensors and return data plus an explicit valid-node mask."""
    if not values:
        raise ValueError("values cannot be empty")
    feature_dim = values[0].shape[-1]
    if any(value.ndim != 2 or value.shape[-1] != feature_dim for value in values):
        raise ValueError("all values must have shape [nodes, shared_features]")
    maximum = max(value.shape[0] for value in values)
    width = maximum if width is None else width
    if width < maximum:
        raise ValueError("width cannot truncate valid nodes")
    result = values[0].new_zeros(len(values), width, feature_dim)
    valid = torch.zeros(len(values), width, dtype=torch.bool, device=values[0].device)
    for index, value in enumerate(values):
        result[index, : value.shape[0]] = value
        valid[index, : value.shape[0]] = True
    return result, valid


class VariableDofInterventionCore(nn.Module):
    """Shared serial-chain transition with exact analytic lock projection."""

    def __init__(self, hidden_dim: int = 96, message_steps: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        # q, qvel, action, locked, normalized depth, local axis(3), origin(3)
        self.node_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        # source code, signed direction and whether source is a fixed relay.
        self.edge_message = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )

    @staticmethod
    def project(
        joint_state: torch.Tensor,
        action: torch.Tensor,
        lock_mask: torch.Tensor,
        lock_angle: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_state.shape[-1] != 2:
            raise ValueError("joint_state must be [batch, nodes, (q, qvel)]")
        expected = joint_state.shape[:2]
        if any(x.shape != expected for x in (action, lock_mask, lock_angle, valid)):
            raise ValueError("action/masks/angles must match [batch, nodes]")
        valid_f = valid.to(joint_state.dtype)
        locked = lock_mask.to(joint_state.dtype) * valid_f
        projected = joint_state.clone()
        projected[..., 0] = (
            projected[..., 0] * (1.0 - locked) + lock_angle * locked
        ) * valid_f
        projected[..., 1] = projected[..., 1] * (1.0 - locked) * valid_f
        return projected, action * (1.0 - locked) * valid_f

    def encode_nodes(
        self,
        joint_state: torch.Tensor,
        action: torch.Tensor,
        lock_mask: torch.Tensor,
        lock_angle: torch.Tensor,
        valid: torch.Tensor,
        axes: torch.Tensor,
        origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, projected_action = self.project(
            joint_state, action, lock_mask, lock_angle, valid
        )
        batch, nodes, _ = state.shape
        if axes.shape != (batch, nodes, 3) or origins.shape != axes.shape:
            raise ValueError("axes/origins must have shape [batch, nodes, 3]")
        valid_f = valid.to(state.dtype)
        count = valid_f.sum(dim=1, keepdim=True).clamp_min(2.0)
        indices = torch.arange(nodes, device=state.device, dtype=state.dtype)[None]
        depth = indices / (count - 1.0)
        features = torch.cat([
            state, projected_action.unsqueeze(-1), lock_mask.unsqueeze(-1),
            depth.unsqueeze(-1), axes, origins,
        ], dim=-1)
        hidden = self.node_encoder(features) * valid_f.unsqueeze(-1)
        for _ in range(self.message_steps):
            messages = torch.zeros_like(hidden)
            for left in range(nodes - 1):
                right = left + 1
                edge_valid = (valid[:, left] & valid[:, right]).to(state.dtype).unsqueeze(-1)
                for source, target, direction in ((left, right, 1.0), (right, left, -1.0)):
                    edge = torch.cat([
                        hidden[:, source],
                        state.new_full((batch, 1), direction),
                        lock_mask[:, source].unsqueeze(-1),
                    ], dim=-1)
                    messages[:, target] += self.edge_message(edge) * edge_valid
            hidden = self.update(
                messages.reshape(-1, self.hidden_dim), hidden.reshape(-1, self.hidden_dim)
            ).view(batch, nodes, self.hidden_dim)
            hidden = hidden * valid_f.unsqueeze(-1)
        return state, projected_action, hidden

    def forward(
        self,
        joint_state: torch.Tensor,
        action: torch.Tensor,
        lock_mask: torch.Tensor,
        lock_angle: torch.Tensor,
        valid: torch.Tensor,
        axes: torch.Tensor,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        state, projected_action, hidden = self.encode_nodes(
            joint_state, action, lock_mask, lock_angle, valid, axes, origins
        )
        valid_f = valid.to(state.dtype)
        delta = self.head(hidden) * (1.0 - lock_mask).unsqueeze(-1) * valid_f.unsqueeze(-1)
        prediction = state + delta
        prediction, _ = self.project(
            prediction, projected_action, lock_mask, lock_angle, valid
        )
        return prediction
