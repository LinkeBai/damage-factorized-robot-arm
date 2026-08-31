import numpy as np
import pytest
import torch

from robotarm.training.decision_focused import (
    PairedCandidateBatch,
    candidate_metrics,
    kendall_correlation,
    load_sequence_candidate_npz,
    paired_soft_regret_loss,
    rollout_candidate_terminal_object,
    spearman_correlation,
    world_model_candidate_metrics,
    world_model_six_stage_metrics,
    world_model_paired_regret_loss,
    world_model_paired_regret_loss_batched,
)


class AdditiveModel:
    def step(self, state, action, mask, angle, hidden=None):
        del mask, angle
        result = state.clone()
        result[:, 10:12] = result[:, 10:12] + action[:, :2]
        return result, hidden


def test_aligned_prediction_has_lower_soft_regret_than_reversed_prediction():
    truth = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    goal = torch.tensor([[0.0, 0.0]])
    aligned = truth.clone()
    reversed_prediction = truth.flip(1)
    assert paired_soft_regret_loss(aligned, truth, goal) < paired_soft_regret_loss(
        reversed_prediction, truth, goal
    )


def test_soft_regret_has_gradient_through_world_model_prediction():
    prediction = torch.tensor([[[0.2, 0.0], [0.1, 0.0]]], requires_grad=True)
    truth = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    loss = paired_soft_regret_loss(prediction, truth, torch.zeros(1, 2))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0


def test_candidate_metrics_detect_perfect_and_reversed_order():
    truth = np.array([0.0, 1.0, 2.0])
    assert spearman_correlation(truth, truth) == pytest.approx(1.0)
    assert kendall_correlation(truth, truth) == pytest.approx(1.0)
    assert spearman_correlation(truth[::-1], truth) == pytest.approx(-1.0)
    assert kendall_correlation(truth[::-1], truth) == pytest.approx(-1.0)
    metrics = candidate_metrics(np.array([2.0, 1.0, 0.0]), truth)
    assert metrics["top1_regret"] == pytest.approx(2.0)


def test_soft_regret_rejects_unmatched_candidate_tensors():
    with pytest.raises(ValueError):
        paired_soft_regret_loss(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2), torch.zeros(1, 2))


def test_grouped_world_model_rollout_preserves_candidate_pairing():
    actions = torch.zeros(1, 3, 2, 5)
    actions[0, :, :, 0] = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    batch = PairedCandidateBatch(
        initial_state=torch.zeros(1, 14),
        actions=actions,
        true_terminal_object=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        goal=torch.zeros(1, 2),
        lock_mask=torch.zeros(1, 5),
        lock_angle=torch.zeros(1, 5),
    )
    prediction = rollout_candidate_terminal_object(AdditiveModel(), batch)
    assert torch.allclose(prediction, batch.true_terminal_object)
    assert world_model_paired_regret_loss(AdditiveModel(), batch) < 1e-6
    metrics = world_model_candidate_metrics(AdditiveModel(), batch)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["top1_regret"] == pytest.approx(0.0)
    assert metrics["groups"] == 1


def test_batched_regret_loss_matches_unbatched_loss():
    actions = torch.zeros(5, 3, 2, 5)
    actions[:, :, :, 0] = torch.tensor([0.0, 0.5, 1.0])[None, :, None]
    truth = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]).expand(5, -1, -1)
    batch = PairedCandidateBatch(
        initial_state=torch.zeros(5, 14), actions=actions,
        true_terminal_object=truth, goal=torch.zeros(5, 2),
        lock_mask=torch.zeros(5, 5), lock_angle=torch.zeros(5, 5),
    )
    expected = world_model_paired_regret_loss(AdditiveModel(), batch)
    actual = world_model_paired_regret_loss_batched(
        AdditiveModel(), batch, group_batch_size=2)
    torch.testing.assert_close(actual, expected)


def test_candidate_batch_rejects_unmatched_initial_states():
    batch = PairedCandidateBatch(
        initial_state=torch.zeros(2, 14), actions=torch.zeros(1, 3, 2, 5),
        true_terminal_object=torch.zeros(1, 3, 2), goal=torch.zeros(1, 2),
        lock_mask=torch.zeros(1, 5), lock_angle=torch.zeros(1, 5),
    )
    with pytest.raises(ValueError):
        batch.validate()


def test_all_split_keeps_every_permitted_group(tmp_path):
    candidates = 4
    groups = np.repeat(np.arange(3), candidates)
    locks = np.repeat(np.array([1, 3, 2]), candidates)
    initial = np.zeros((12, 14), dtype=np.float32)
    initial[:, :5] = np.arange(5, dtype=np.float32)
    actions = np.zeros((12, 5, 5), dtype=np.float32)
    goals = np.repeat(np.array([[0.2, 0.1]], dtype=np.float32), 12, axis=0)
    states = np.zeros((12, 5, 14), dtype=np.float32)
    episodes = np.repeat(np.arange(3), candidates)
    path = tmp_path / "all.npz"
    np.savez(path, group=groups, locked_joint=locks, initial_state=initial,
             action_sequence=actions, goal=goals, segment_states=states,
             episode=episodes)
    batch = load_sequence_candidate_npz(
        path, allowed_locked_joints=(1, 3), split="all", segment_repeat=10)
    assert batch.actions.shape == (2, candidates, 50, 5)
    assert torch.equal(batch.lock_mask.argmax(dim=1), torch.tensor([1, 3]))


def test_confirmation_loader_requires_explicit_d3_permission(tmp_path):
    candidates = 4
    path = tmp_path / "d3_confirmation.npz"
    np.savez(
        path,
        group=np.repeat(np.arange(2), candidates),
        episode=np.repeat(np.arange(2), candidates),
        locked_joint=np.full(2 * candidates, 2, dtype=np.int64),
        initial_state=np.zeros((2 * candidates, 14), dtype=np.float32),
        action_sequence=np.zeros((2 * candidates, 5, 5), dtype=np.float32),
        goal=np.zeros((2 * candidates, 2), dtype=np.float32),
        segment_states=np.zeros((2 * candidates, 5, 14), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="no candidate groups remain"):
        load_sequence_candidate_npz(path, split="all", segment_repeat=10)
    batch = load_sequence_candidate_npz(
        path, allowed_locked_joints=(2,), split="all", segment_repeat=10)
    assert batch.actions.shape == (2, candidates, 50, 5)
    assert torch.equal(batch.lock_mask.argmax(dim=1), torch.tensor([2, 2]))


def test_six_stage_metrics_use_same_selected_candidate_and_labels():
    actions = torch.zeros(1, 3, 2, 5)
    actions[0, :, :, 0] = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    batch = PairedCandidateBatch(
        initial_state=torch.zeros(1, 14), actions=actions,
        true_terminal_object=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        goal=torch.zeros(1, 2), lock_mask=torch.zeros(1, 5),
        lock_angle=torch.zeros(1, 5),
        true_contact=torch.tensor([[[False, False], [True, True], [True, True]]]),
        true_success=torch.tensor([[True, False, False]]),
        true_min_contact_distance=torch.tensor([[
            [0.02, 0.02], [0.0, 0.0], [0.0, 0.0]
        ]]),
    )
    metrics = world_model_six_stage_metrics(AdditiveModel(), batch)
    assert metrics["groups"] == 1
    assert metrics["action_ranking"]["spearman"] == pytest.approx(1.0)
    assert metrics["contact"]["selected_candidate_rate"] == pytest.approx(0.0)
    assert metrics["closed_loop_outcome"]["success_rate"] == pytest.approx(1.0)
    assert metrics["closed_loop_outcome"]["endpoint_error"] == pytest.approx(0.0)
    assert metrics["response"]["contact_candidate_count"] == 2
    assert metrics["reachability"]["realized_near_contact_rate"] == pytest.approx(0.0)


def test_loader_preserves_optional_contact_and_success_labels(tmp_path):
    candidates = 2
    path = tmp_path / "labels.npz"
    np.savez(
        path,
        group=np.repeat(np.arange(2), candidates),
        episode=np.repeat(np.arange(2), candidates),
        locked_joint=np.repeat(np.array([1, 3]), candidates),
        initial_state=np.zeros((4, 14), dtype=np.float32),
        action_sequence=np.zeros((4, 5, 5), dtype=np.float32),
        goal=np.zeros((4, 2), dtype=np.float32),
        segment_states=np.zeros((4, 5, 14), dtype=np.float32),
        contact_by_segment=np.array([[0, 1, 0, 0, 0]] * 4, dtype=np.int8),
        success=np.array([1, 0, 0, 1], dtype=np.int8),
        minimum_contact_distance_by_segment=np.zeros((4, 5), dtype=np.float32),
    )
    batch = load_sequence_candidate_npz(path, split="all", segment_repeat=2)
    assert batch.true_contact is not None
    assert batch.true_contact.shape == (2, 2, 10)
    assert batch.true_success is not None
    assert batch.true_success.tolist() == [[True, False], [False, True]]
    assert batch.true_min_contact_distance is not None
    assert batch.true_min_contact_distance.shape == (2, 2, 10)
