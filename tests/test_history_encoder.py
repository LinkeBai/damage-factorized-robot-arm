from __future__ import annotations

import torch

from robotarm.models.history_encoder import HistoryEncoder


def test_history_encoder_output_shape():
    encoder = HistoryEncoder(state_dim=10, action_dim=5, out_dim=8)
    states = torch.zeros(3, 8, 10)  # K=3 trajectories, T=8 steps
    actions = torch.zeros(3, 8, 5)
    context = encoder(states, actions)
    assert context.shape == (8,)


def test_history_encoder_single_trajectory_broadcasts():
    encoder = HistoryEncoder(state_dim=10, action_dim=5, out_dim=8)
    # 2-D inputs (no batch dim) should be promoted to a single-trajectory batch.
    states = torch.zeros(8, 10)
    actions = torch.zeros(8, 5)
    context = encoder(states, actions)
    assert context.shape == (8,)


def test_history_encoder_is_deterministic_given_input():
    encoder = HistoryEncoder(state_dim=10, action_dim=5, out_dim=8).eval()
    states = torch.randn(2, 8, 10)
    actions = torch.randn(2, 8, 5)
    with torch.no_grad():
        c1 = encoder(states, actions)
        c2 = encoder(states, actions)
    assert torch.allclose(c1, c2)


def test_history_encoder_parameter_count_matches_dfwm_order():
    # Fairness: the encoder alone should be a small sequence encoder, not
    # dramatically larger than the topology encoder it complements.
    encoder = HistoryEncoder(state_dim=10, action_dim=5, hidden_dim=64, out_dim=8)
    n_params = sum(p.numel() for p in encoder.parameters())
    # GRU (15->64) + linear head (64->8) is a small sequence encoder; the full
    # DFWM method is ~149k (encoder + world model).
    assert 0 < n_params < 30_000
