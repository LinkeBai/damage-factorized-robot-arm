from robotarm.hardware.push_return_fsm import CycleState, Observation, PushReturnFSM


def test_complete_one_guarded_cycle_under_budget():
    fsm = PushReturnFSM(10.0)
    observations = [
        Observation(start_stable=True), Observation(),
        Observation(forward_finished=True, elapsed_motion_s=2.5),
        Observation(outcome_available=True, elapsed_motion_s=2.5),
        Observation(retract_finished=True, elapsed_motion_s=3.3),
        Observation(behind_clearance_ok=True, elapsed_motion_s=4.7),
        Observation(descend_finished=True, elapsed_motion_s=5.4),
        Observation(reset_stable=True, elapsed_motion_s=8.2),
        Observation(home_finished=True, elapsed_motion_s=9.4),
        Observation(audit_passed=True, elapsed_motion_s=9.4),
    ]
    for observation in observations:
        fsm.advance(observation)
    assert fsm.state is CycleState.COMPLETE
    assert fsm.abort_reason is None


def test_time_budget_is_hard_abort():
    fsm = PushReturnFSM(10.0)
    assert fsm.advance(Observation(elapsed_motion_s=10.01)) is CycleState.ABORT
    assert fsm.abort_reason == "motion_time_budget_exceeded"


def test_any_camera_loss_is_hard_abort():
    fsm = PushReturnFSM()
    assert fsm.advance(Observation(cube_detected=False)) is CycleState.ABORT
    assert fsm.abort_reason == "cube_observation_lost"


def test_person_entering_after_start_is_hard_abort():
    fsm = PushReturnFSM()
    fsm.advance(Observation(start_stable=True))
    assert fsm.advance(Observation(scene_clear=False)) is CycleState.ABORT
    assert fsm.abort_reason == "scene_not_clear"

