import numpy as np
import torch
from torch import nn

import scripts.prepare_real_ipwm_trial as preparation
from scripts.prepare_real_ipwm_trial import score_reference, score_references_batched
from scripts.prepare_real_ipwm_trial import score_references_terminal_only


def test_batched_gpu_ready_fk_matches_scalar_fk():
    rng = np.random.default_rng(20260903)
    q = rng.uniform(-1.0, 1.0, size=(16, 5))
    actual = preparation.forward_kinematics_batched(
        torch.as_tensor(q, dtype=torch.float64)
    ).numpy()
    expected = np.stack([preparation.forward_kinematics(row) for row in q])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_batched_pose_matches_scalar_pose():
    rng = np.random.default_rng(20260904)
    q = rng.uniform(-1.0, 1.0, size=(16, 5))
    actual = preparation.forward_pose_batched(
        torch.as_tensor(q, dtype=torch.float64)
    ).numpy()
    expected = np.stack([preparation.forward_pose(row) for row in q])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_contact_geometry_gate_detects_tilt_and_height_drift():
    q0 = np.array([0.1, 0.7, 0.5, 0.6, -0.2])
    qrefs = np.repeat(q0[None, None, :], 2, axis=0)
    qrefs = np.repeat(qrefs, 5, axis=1)
    qrefs[1, :, 3] += np.linspace(0.0, 0.4, 5)
    metrics = preparation.contact_geometry_metrics(qrefs, np.array([1.0, 0.0]))
    assert metrics["maximum_face_rotation_deg"][0] == 0.0
    assert metrics["maximum_height_deviation_m"][0] == 0.0
    assert metrics["maximum_face_rotation_deg"][1] > 10.0
    assert metrics["maximum_axis_alignment_error_deg"].shape == (2,)
    assert metrics["maximum_height_deviation_m"][1] > 0.0


def test_batched_ik_holds_multiple_locked_axes_exactly():
    q0 = np.array([0.1, 0.7, 0.5, 0.6, -0.2])
    targets = np.stack([
        preparation.forward_kinematics(q0) + np.array([delta, 0.0, 0.0])
        for delta in (0.005, 0.01, 0.015)
    ])
    ranges = np.tile(np.array([[-1.4, 1.4]]), (5, 1))
    solved, errors = preparation.inverse_kinematics_batched(
        targets, ranges, q0, (0, 4), torch.device("cpu"), max_steps=250,
    )
    np.testing.assert_allclose(solved[:, 0], q0[0], atol=0, rtol=0)
    np.testing.assert_allclose(solved[:, 4], q0[4], atol=0, rtol=0)
    assert np.all(errors < 0.005)


class DeterministicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def step(self, state, action, mask, angle, hidden):
        next_state = state.clone()
        next_state[:, :5] = next_state[:, :5] + 0.01 * action
        next_state[:, 5:10] = action
        next_state[:, 10:12] = next_state[:, 10:12] + 0.002 * action[:, :2]
        next_state[:, :5] = next_state[:, :5] * (1 - mask) + angle * mask
        return next_state, None


def test_batched_scoring_matches_individual_scoring():
    rng = np.random.default_rng(7)
    model = DeterministicModel().eval()
    initial = np.r_[np.zeros(10), [.2, .1, 0, 0]]
    qrefs = rng.normal(0, .03, (7, 9, 5))
    goal = np.asarray([.23, .1])
    mask = np.asarray([0, 1, 0, 0, 0], dtype=float)
    angle = np.zeros(5)
    expected = [score_reference(model, initial, qref, goal, mask, angle) for qref in qrefs]
    scores, actions, predictions = score_references_batched(
        model, initial, qrefs, goal, mask, angle, batch_size=3,
    )
    np.testing.assert_allclose(scores, [row[0] for row in expected], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actions, np.stack([row[1] for row in expected]), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(predictions, np.stack([row[2] for row in expected]), rtol=1e-6, atol=1e-7)
    online_scores = score_references_terminal_only(
        model, initial, qrefs, goal, mask, angle, batch_size=3,
    )
    np.testing.assert_allclose(online_scores, scores, rtol=1e-6, atol=1e-7)
