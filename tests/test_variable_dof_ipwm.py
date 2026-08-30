from pathlib import Path

import numpy as np
import torch

from robotarm.models.variable_dof_ipwm import SerialChainSpec, VariableDofInterventionCore


ROOT = Path(__file__).resolve().parents[1]


def _geometry(spec: SerialChainSpec, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    axes = torch.zeros(1, width, 3)
    origins = torch.zeros(1, width, 3)
    axes[0, : spec.dof] = torch.from_numpy(spec.axes)
    origins[0, : spec.dof] = torch.from_numpy(spec.origins)
    return axes, origins


def test_same_parameters_accept_full_five_and_seven_dof_models():
    genki = SerialChainSpec.from_mjcf(
        ROOT / "sim/assets/genkiarm_push.xml",
        tuple(f"j{i}" for i in range(1, 6)), name="genkiarm",
    )
    panda = SerialChainSpec.from_mjcf(
        ROOT / "sim/assets/panda_push_grasp.xml",
        tuple(f"joint{i}" for i in range(1, 8)), name="panda",
    )
    assert genki.dof == 5 and panda.dof == 7
    assert not np.allclose(genki.axes[:5], panda.axes[:5])
    model = VariableDofInterventionCore(hidden_dim=16)
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    for spec in (genki, panda):
        state = torch.zeros(1, spec.dof, 2)
        action = torch.zeros(1, spec.dof)
        mask = torch.zeros_like(action)
        angle = torch.zeros_like(action)
        valid = torch.ones_like(action, dtype=torch.bool)
        axes, origins = _geometry(spec, spec.dof)
        output = model(state, action, mask, angle, valid, axes, origins)
        assert output.shape == state.shape
        assert {id(parameter) for parameter in model.parameters()} == parameter_ids


def test_projection_is_exact_for_locked_nodes_and_padding_is_invariant():
    torch.manual_seed(7)
    spec = SerialChainSpec.from_mjcf(
        ROOT / "sim/assets/genkiarm_push.xml",
        tuple(f"j{i}" for i in range(1, 6)), name="genkiarm",
    )
    model = VariableDofInterventionCore(hidden_dim=16).eval()
    state5 = torch.randn(1, 5, 2)
    action5 = torch.randn(1, 5)
    mask5 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    angle5 = torch.tensor([[0.0, 0.0, 0.37, 0.0, 0.0]])
    valid5 = torch.ones(1, 5, dtype=torch.bool)
    axes5, origins5 = _geometry(spec, 5)
    out5 = model(state5, action5, mask5, angle5, valid5, axes5, origins5)
    assert out5[0, 2, 0].item() == angle5[0, 2].item()
    assert out5[0, 2, 1].item() == 0.0

    state7 = torch.cat([state5, torch.randn(1, 2, 2)], dim=1)
    action7 = torch.cat([action5, torch.randn(1, 2)], dim=1)
    mask7 = torch.cat([mask5, torch.ones(1, 2)], dim=1)
    angle7 = torch.cat([angle5, torch.randn(1, 2)], dim=1)
    valid7 = torch.tensor([[True, True, True, True, True, False, False]])
    axes7, origins7 = _geometry(spec, 7)
    out7 = model(state7, action7, mask7, angle7, valid7, axes7, origins7)
    torch.testing.assert_close(out7[:, :5], out5, rtol=0, atol=1e-7)
    torch.testing.assert_close(out7[:, 5:], torch.zeros_like(out7[:, 5:]))
