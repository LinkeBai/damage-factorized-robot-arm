from __future__ import annotations

import torch

from robotarm.models.topology_surgery import TopologySurgery


def test_projects_only_locked_arm_dimensions() -> None:
    surgery = TopologySurgery()
    state = torch.arange(28, dtype=torch.float32).view(2, 14)
    mask = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    angle = torch.tensor([[0, 0.5, 0, 0, 0], [0, 0, -0.5, 0, 0]], dtype=torch.float32)
    projected = surgery.project_state(state, mask, angle)

    assert projected[0, 1].item() == 0.5
    assert projected[0, 6].item() == 0.0
    assert projected[1, 2].item() == -0.5
    assert projected[1, 7].item() == 0.0
    assert torch.equal(projected[:, 10:], state[:, 10:])
    assert torch.equal(projected[:, [0, 3, 4]], state[:, [0, 3, 4]])


def test_projects_locked_actions_and_reports_zero_violation() -> None:
    surgery = TopologySurgery()
    mask = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
    angle = torch.tensor([[0, 0.5, 0, 0, 0]], dtype=torch.float32)
    state = torch.randn(1, 14)
    projected = surgery.project_state(state, mask, angle)
    action = surgery.project_action(torch.ones(1, 5), mask)

    assert torch.equal(action, torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]]))
    assert torch.allclose(surgery.constraint_violation(projected, mask, angle), torch.zeros(1))
