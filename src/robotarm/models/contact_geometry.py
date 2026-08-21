"""Differentiable, parameter-free pusher/box contact geometry."""
from __future__ import annotations

import torch


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis
    zero = x * 0
    skew = torch.stack((torch.stack((zero, -z, y)),
                        torch.stack((z, zero, -x)),
                        torch.stack((-y, x, zero))))
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis[:, None] * axis[None, :]
    return (torch.cos(angle)[..., None, None] * eye
            + (1.0 - torch.cos(angle))[..., None, None] * outer
            + torch.sin(angle)[..., None, None] * skew)


def pusher_box_contact_gate(q: torch.Tensor, block_xy: torch.Tensor, *,
                            threshold: float = -0.005,
                            temperature: float = 0.002) -> torch.Tensor:
    """Return a soft contact indicator from analytic capsule/box separation."""
    batch = q.shape[0]
    axes = q.new_tensor([[0., 0., 1.], [0., 1., 0.], [0., 1., 0.],
                         [0., 1., 0.], [0., 0., 1.]])
    origins = q.new_tensor([[0., 0., .120], [0., 0., 0.], [0., 0., .110],
                            [0., 0., .120], [0., 0., .060]])
    rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch, 3, 3).clone()
    position = torch.zeros(batch, 3, device=q.device, dtype=q.dtype)
    for joint in range(q.shape[1]):
        origin = origins[joint].view(1, 3, 1).expand(batch, -1, -1)
        position = position + torch.bmm(rotation, origin).squeeze(-1)
        rotation = torch.bmm(rotation, _axis_rotation(axes[joint], q[:, joint]))
    tool = position + torch.bmm(
        rotation, q.new_tensor([0.0, -0.0132, 0.110]).view(1, 3, 1).expand(batch, -1, -1)
    ).squeeze(-1)
    local = q.new_tensor([[0.0, 0.0, 0.0], [0.0200, 0.0, 0.0],
                          [0.0537, -0.0210, 0.0210]])
    points = torch.stack([
        tool + torch.bmm(rotation, point.view(1, 3, 1).expand(batch, -1, -1)).squeeze(-1)
        for point in local
    ], dim=1)
    block_xyz = torch.cat((block_xy, block_xy.new_full((batch, 1), 0.02)), dim=-1)
    lower, upper = block_xyz - 0.02, block_xyz + 0.02
    gaps = []
    for index, radius in ((0, 0.012), (1, 0.008)):
        first, direction = points[:, index], points[:, index + 1] - points[:, index]
        safe = torch.where(direction.abs() > 1e-8, direction, torch.ones_like(direction))
        fractions = torch.stack((
            torch.zeros(batch, device=q.device, dtype=q.dtype),
            torch.ones(batch, device=q.device, dtype=q.dtype),
            (lower[:, 0] - first[:, 0]) / safe[:, 0],
            (upper[:, 0] - first[:, 0]) / safe[:, 0],
            (lower[:, 1] - first[:, 1]) / safe[:, 1],
            (upper[:, 1] - first[:, 1]) / safe[:, 1],
            (lower[:, 2] - first[:, 2]) / safe[:, 2],
            (upper[:, 2] - first[:, 2]) / safe[:, 2],
            ((block_xyz - first) * direction).sum(-1)
            / direction.pow(2).sum(-1).clamp_min(1e-8),
        ), dim=1).clamp(0.0, 1.0)
        segment_points = first[:, None, :] + fractions[..., None] * direction[:, None, :]
        box_points = torch.maximum(torch.minimum(segment_points, upper[:, None, :]),
                                   lower[:, None, :])
        distance = torch.linalg.vector_norm(
            box_points - segment_points, dim=-1
        ).min(dim=1).values
        gaps.append(distance - radius)
    gap = torch.stack(gaps, dim=1).min(dim=1).values
    return torch.sigmoid((threshold - gap) / temperature)
