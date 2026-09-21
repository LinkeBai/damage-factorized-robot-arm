import numpy as np

from robotarm.deployment.ipwm_receding_horizon import IPWMRecedingHorizonPlanner


def _score(initial, candidates, goal, mask, angle):
    # Deterministic stand-in for IPWM: prefer terminal J3 displacement matching
    # the currently observed remaining object-to-goal distance.
    desired = float(np.linalg.norm(goal - initial[10:12]))
    moved = np.abs(candidates[:, -1, 2] - initial[2])
    return np.abs(moved - desired * 10.0)


def _planner():
    deltas = np.zeros((10_000, 5, 5), dtype=float)
    terminal = np.linspace(0.0, 0.6, len(deltas))
    for depth in range(5):
        deltas[:, depth, 2] = terminal * (depth + 1) / 5
        deltas[:, depth, 3] = -deltas[:, depth, 2]
    return IPWMRecedingHorizonPlanner(
        deltas,
        task_start_px=(100.0, 50.0), task_goal_px=(130.0, 50.0),
        locked_indices=(0, 1), score_function=_score,
        metres_per_pixel=0.001,
        base_xy_per_pixel=np.array([0.001, 0.0]),
        base_xy_intercept_m=np.zeros(2),
    )


def test_true_replan_changes_command_after_new_object_observation():
    planner = _planner()
    common = dict(joint_q=np.zeros(5), joint_qd=np.zeros(5),
                  lock_angles=np.array([0.1, -0.2, 0, 0, 0]))
    first = planner.plan(object_px=(100.0, 50.0), observation_monotonic_ns=1, **common)
    second = planner.plan(object_px=(124.0, 50.0), observation_monotonic_ns=2, **common)
    assert first.cycle == 0 and second.cycle == 1
    assert first.selected_index != second.selected_index
    assert not np.allclose(first.selected_first_reference, second.selected_first_reference)


def test_locked_joints_and_candidate_count_are_invariant():
    planner = _planner()
    plan = planner.plan(
        joint_q=np.zeros(5), joint_qd=np.zeros(5), object_px=(110.0, 50.0),
        observation_monotonic_ns=3,
        lock_angles=np.array([0.3, -0.4, 0, 0, 0]),
    )
    assert plan.scores.shape == (10_000,)
    assert np.all(plan.selected_references[:, 0] == 0.3)
    assert np.all(plan.selected_references[:, 1] == -0.4)
    assert len(plan.candidates_sha256) == 64


def test_current_state_safety_filter_excludes_out_of_range_candidate():
    planner = _planner()
    planner.joint_ranges_rad = np.array([[-1, 1], [-1, 1], [-0.2, 0.2], [-1, 1], [-1, 1]])
    plan = planner.plan(
        joint_q=np.zeros(5), joint_qd=np.zeros(5), object_px=(100.0, 50.0),
        observation_monotonic_ns=4, lock_angles=np.zeros(5),
    )
    assert abs(plan.selected_references[-1, 2]) <= 0.2


def test_minimum_amplitude_preserves_full_rescoring_and_locks_near_goal():
    planner = _planner()
    planner.minimum_remaining_fraction = 0.5
    plan = planner.plan(joint_q=np.zeros(5), joint_qd=np.zeros(5),
                        object_px=(129., 50.), observation_monotonic_ns=1,
                        lock_angles=np.array([.1,-.2,0,0,0]))
    assert plan.remaining_fraction == 0.5
    assert plan.scores.shape == (10000,)
    assert np.allclose(plan.candidates[:,:,2], .5 * planner.reference_deltas[:,:,2])
    assert np.all(plan.candidates[:,:,0] == .1)
    assert np.all(plan.candidates[:,:,1] == -.2)


def test_affine_mapping_preserves_cross_axis_error_in_model_input():
    recorded = []
    def score(initial, candidates, goal, mask, angle):
        recorded.append((initial.copy(), goal.copy()))
        return np.zeros(len(candidates))
    affine = np.array([[.001, .0002, -.1], [-.0003, .002, .4]])
    planner = IPWMRecedingHorizonPlanner(
        np.zeros((10000, 2, 5)), task_start_px=(100.,50.),
        task_goal_px=(130.,50.), locked_indices=(), score_function=score,
        metres_per_pixel=.001, base_xy_per_pixel=np.array([.001,0.]),
        base_xy_intercept_m=np.zeros(2), pixel_to_base_affine=affine)
    for y in (50., 55.):
        planner.plan(joint_q=np.zeros(5), joint_qd=np.zeros(5),
                     object_px=(110.,y), observation_monotonic_ns=int(y),
                     lock_angles=np.zeros(5))
    assert np.allclose(recorded[1][0][10:12]-recorded[0][0][10:12], [.001,.01])
    assert np.allclose(recorded[0][1], affine @ [130.,50.,1.])
    assert np.array_equal(recorded[0][1], recorded[1][1])


def test_affine_mapping_rejects_degenerate_or_nonfinite_matrix():
    import pytest
    for affine in (np.zeros((2,3)), np.ones((3,2)), np.full((2,3), np.nan)):
        with pytest.raises(ValueError, match='pixel_to_base_affine'):
            IPWMRecedingHorizonPlanner(
                np.zeros((2,2,5)), task_start_px=(0.,0.),task_goal_px=(1.,0.),
                locked_indices=(), score_function=_score, metres_per_pixel=.001,
                base_xy_per_pixel=np.array([.001,0.]),base_xy_intercept_m=np.zeros(2),
                pixel_to_base_affine=affine)
