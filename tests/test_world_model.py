"""Tests for the world model + residual latent optimization (M2/M3)."""
from __future__ import annotations

import torch

from robotarm.models.residual_context import (
    LatentOptConfig,
    compose_context,
    latent_optimize,
)
from robotarm.models.world_model import WorldModel, WorldModelConfig

STATE, ACT = 12, 6
TOPO, RES = 8, 4
CTX = TOPO + RES


def make_wm():
    return WorldModel(WorldModelConfig(state_dim=STATE, action_dim=ACT, context_dim=CTX))


def make_data(B=4, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(B, T, STATE, generator=g)
    actions = torch.randn(B, T, ACT, generator=g)
    return states, actions


def test_wm_step_shapes():
    wm = make_wm()
    st = torch.randn(2, STATE)
    ac = torch.randn(2, ACT)
    ctx = torch.randn(2, CTX)
    pred, h = wm.step(st, ac, ctx, None)
    assert pred["mean"].shape == (2, STATE)
    assert pred["log_std"].shape == (2, STATE)
    assert pred["reward"].shape == (2,)
    assert pred["continue"].shape == (2,)
    assert h.shape == (2, wm.cfg.latent_dim)


def test_wm_nll_finite():
    wm = make_wm()
    st = torch.randn(STATE)
    ac = torch.randn(ACT)
    ctx = torch.randn(CTX)
    pred, _ = wm.step(st.unsqueeze(0), ac.unsqueeze(0), ctx.unsqueeze(0), None)
    nll = wm.nll(pred, torch.randn(1, STATE))
    assert torch.isfinite(nll).all()
    assert nll.shape == (1,)


def test_rssm_observe_step_has_prior_posterior_and_kl():
    wm = make_wm()
    state = torch.randn(2, STATE)
    action = torch.randn(2, ACT)
    next_state = torch.randn(2, STATE)
    context = torch.randn(2, CTX)
    posterior, prior, hidden = wm.observe_step(
        state, action, next_state, context, None
    )
    assert posterior["posterior_mean"].shape == (
        2,
        wm.cfg.stochastic_dim,
    )
    assert posterior["prior_mean"].shape == (2, wm.cfg.stochastic_dim)
    assert posterior["kl"].shape == (2,)
    assert torch.all(posterior["kl"] >= -1e-6)
    assert prior["mean"].shape == (2, STATE)
    assert hidden.shape == (2, wm.cfg.latent_dim)


def test_predict_observed_nll():
    wm = make_wm()
    states, actions = make_data(B=1, T=6)
    out = wm.predict_multi_step(states[0], actions[0], torch.randn(CTX))
    assert out["nll"].shape == (5,)  # T-1

def test_predict_future_rollout():
    wm = make_wm()
    states, actions = make_data(B=1, T=4)
    out = wm.predict_multi_step(states[0], actions[0], torch.randn(CTX), n_future=3)
    assert out["future_mean"].shape == (3, STATE)


def test_latent_optimize_fit_improves():
    """On synthetic data, optimizing z should lower WM multi-step NLL vs z=0."""
    wm = make_wm()
    states, actions = make_data(B=3, T=12, seed=1)
    topology = torch.randn(TOPO)
    cfg = LatentOptConfig(d=RES, lr=0.3, steps=60)

    # Baseline NLL with z = 0 (frozen WM).
    base_context = compose_context(topology, torch.zeros(RES), context_dim=CTX)
    with torch.no_grad():
        base = sum(wm.predict_multi_step(states[k], actions[k], base_context)["nll"].sum().item()
                   for k in range(3))
    rc = latent_optimize(wm, topology, states, actions, cfg)
    adapted_context = compose_context(topology, rc.z, context_dim=CTX)
    with torch.no_grad():
        opt = sum(wm.predict_multi_step(states[k], actions[k], adapted_context)["nll"].sum().item()
                  for k in range(3))
    assert opt <= base + 1e-6  # optimized z should not be worse than zero start


def test_latent_optimize_moves_z():
    wm = make_wm()
    states, actions = make_data(B=2, T=8, seed=2)
    rc = latent_optimize(
        wm,
        torch.randn(TOPO),
        states,
        actions,
        LatentOptConfig(d=RES, steps=30, lr=0.3),
    )
    assert rc.z.shape == (RES,)
    assert torch.isnan(rc.z).any().item() is False
    assert not torch.allclose(rc.z, torch.zeros_like(rc.z))


def test_residual_changes_world_model_prediction():
    wm = make_wm()
    state = torch.randn(1, STATE)
    action = torch.randn(1, ACT)
    topology = torch.randn(TOPO)
    c0 = compose_context(topology, torch.zeros(RES), context_dim=CTX)
    c1 = compose_context(topology, torch.ones(RES), context_dim=CTX)
    p0, _ = wm.step(state, action, c0.unsqueeze(0), None)
    p1, _ = wm.step(state, action, c1.unsqueeze(0), None)
    assert not torch.allclose(p0["mean"], p1["mean"])
