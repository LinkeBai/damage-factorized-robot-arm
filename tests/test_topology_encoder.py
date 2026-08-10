"""Tests for the topology encoder (PROJECT-PLAN-V4 §4.2)."""
from __future__ import annotations

import torch

from robotarm.models.topology_encoder import (
    TopologyEncoder,
    TopologyEncoderConfig,
    build_joint_features,
)

DEFAULT_AXES = torch.tensor(
    [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
    dtype=torch.float32,
)
DEFAULT_LIMITS = torch.tensor([
    [-1.5708, 1.5708],  # J1: +/-90 deg
    [-1.3090, 1.3963],  # J2: -75 to 80 deg
    [-1.3090, 1.3963],  # J3: -75 to 80 deg
    [-1.3090, 1.4835],  # J4: -75 to 85 deg
    [-1.5708, 1.5708],  # J5: +/-90 deg
], dtype=torch.float32)
DEPTH = torch.arange(5, dtype=torch.float32)


def make_encoder():
    cfg = TopologyEncoderConfig(dof=5, out_dim=32)
    return TopologyEncoder(cfg)


def test_output_shape_single():
    enc = make_encoder()
    mask = torch.tensor([0, 0, 1, 0, 0])
    lock = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    out = enc(mask, lock, DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    assert out.shape == (32,)


def test_output_shape_batch():
    enc = make_encoder()
    B = 4
    mask = torch.zeros(B, 5)
    mask[:, 2] = 1
    lock = torch.zeros(B, 5)
    lock[:, 2] = 0.5
    axes = DEFAULT_AXES.unsqueeze(0).expand(B, -1, -1)
    limits = DEFAULT_LIMITS.unsqueeze(0).expand(B, -1, -1)
    depth = DEPTH.unsqueeze(0).expand(B, -1)
    out = enc(mask, lock, axes, limits, depth)
    assert out.shape == (B, 32)


def test_discrete_topology_gives_discrete_embedding():
    # Same mask+lock must map to the same embedding (deterministic, no data).
    enc = make_encoder()
    m1 = torch.tensor([0, 0, 1, 0, 0])
    l1 = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    e1 = enc(m1, l1, DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    e2 = enc(m1.clone(), l1.clone(), DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    torch.testing.assert_close(e1, e2)


def test_different_topology_differs():
    enc = make_encoder()
    e_intact = enc(torch.zeros(5), torch.zeros(5), DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    e_d2 = enc(
        torch.tensor([0, 0, 1, 0, 0]),
        torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0]),
        DEFAULT_AXES, DEFAULT_LIMITS, DEPTH,
    )
    assert torch.linalg.norm(e_intact - e_d2) > 1e-6


def test_lock_angle_affects_embedding():
    enc = make_encoder()
    e_a = enc(mask_d2(), lock_at(0.3), DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    e_b = enc(mask_d2(), lock_at(0.9), DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    assert torch.linalg.norm(e_a - e_b) > 1e-6


def test_grad_flows_to_input():
    enc = make_encoder()
    mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32, requires_grad=True)
    out = enc(mask, torch.zeros(5), DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    out.sum().backward()
    assert mask.grad is not None


def test_build_features_shape():
    mask = torch.tensor([0, 0, 1, 0, 0])
    lock = torch.zeros(5)
    f = build_joint_features(mask, lock, DEFAULT_AXES, DEFAULT_LIMITS, DEPTH)
    assert f.shape == (5, 8)


def mask_d2():
    return torch.tensor([0, 0, 1, 0, 0])


def lock_at(v):
    z = torch.zeros(5)
    z[2] = v
    return z
