import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_real_push_fixed_trajectory as runner
from scripts.recover_j5 import ServoBus as RecoverServoBus

from robotarm.deployment.fixed_raw_trajectory import (
    TICKS_PER_DEGREE,
    build_alignment_events,
    interpolate_raw_waypoints,
    load_fixed_raw_trajectory,
    load_safety_envelope,
    minimum_safe_command_interval_s,
    locked_indices_for_condition,
    validate_interpolated_events,
)


def test_all_fault_subset_labels_have_canonical_lock_indices() -> None:
    assert locked_indices_for_condition("D1") == (0,)
    assert locked_indices_for_condition("D5") == (4,)
    assert locked_indices_for_condition("J1+J3+J5") == (0, 2, 4)


def test_multi_lock_trajectory_requires_every_locked_axis_constant(tmp_path: Path) -> None:
    path = tmp_path / "multi.csv"
    write_raw_trajectory(path, condition="J1+J5", second=(2023, 2070, 2058, 2076, 2067))
    safety = load_safety_envelope(SAFETY_PATH)
    with pytest.raises(ValueError, match="locked j5 target"):
        load_fixed_raw_trajectory(
            path, trajectory_id="operator_supplied", condition="J1+J5",
            safety=safety, maximum_speed_deg_s=5.0,
        )
from scripts.capture_real_push_waypoint import (
    FIELDS,
    append_waypoint,
    capture_positions_readonly,
    raw_to_radians,
)
from scripts.audit_level_a_trajectory_library import audit as audit_level_a_library
from scripts.run_real_push_fixed_trajectory import (
    GoalValidationSchedule,
    STATIC_BATCH_SETTLE_S,
    StaticRegisterExpectation,
    TRAJECTORY_GOAL_READ_ATTEMPTS,
    TRAJECTORY_GOAL_READ_RETRY_DELAY_S,
    _next_capture_deadline_ns,
    batch_write_and_verify_static_registers,
    build_damage_activation_record,
    command_changed_trajectory_targets,
    main,
    record_torque_shutdown_report,
    seed_and_enable_servos,
    telemetry_fields,
    torque_off_all,
    torque_off_static_configuration_expectations,
    validate_recorder_cadence,
    validate_trajectory_goal_readback,
    write_u8_verified,
    write_u16_verified,
)


ROOT = Path(__file__).resolve().parents[1]
SAFETY_PATH = ROOT / "hardware/safety_limits.yaml"
CAMERA_SETTINGS = ROOT / "results/real_robot/camera_settings_selected.json"


def write_raw_trajectory(
    path: Path,
    *,
    condition: str = "D2",
    second: tuple[int, int, int, int, int] = (2079, 2066, 2058, 2076, 2066),
) -> None:
    fields = [
        "trajectory_id", "condition", "waypoint_index", "time_s",
        "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw",
    ]
    rows = [
        ["operator_supplied", condition, 0, 0.0, 2023, 2066, 2058, 2076, 2066],
        ["operator_supplied", condition, 1, 1.0, *second],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_raw_loader_and_integer_interpolation_stay_below_limit(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    write_raw_trajectory(path, second=(2079, 2066, 2040, 2090, 2045))
    safety = load_safety_envelope(SAFETY_PATH)
    trajectory = load_fixed_raw_trajectory(
        path,
        trajectory_id="operator_supplied",
        condition="D2",
        safety=safety,
    )
    events = interpolate_raw_waypoints(trajectory)
    assert events[0].targets_raw == trajectory.waypoints[0].targets_raw
    assert events[-1].targets_raw == trajectory.waypoints[-1].targets_raw
    assert all(event.targets_raw[1] == 2066 for event in events)
    assert max(
        max(abs(after - before) for after, before in zip(b.targets_raw, a.targets_raw))
        for a, b in zip(events, events[1:])
    ) <= 1
    assert validate_interpolated_events(events, safety, 5.0) <= 5.0 + 1e-8


def test_d2_locked_target_change_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "locked.csv"
    write_raw_trajectory(path, second=(2023, 2067, 2058, 2076, 2066))
    with pytest.raises(ValueError, match="locked j2 target"):
        load_fixed_raw_trajectory(
            path,
            trajectory_id="operator_supplied",
            condition="D2",
            safety=load_safety_envelope(SAFETY_PATH),
        )


def test_overspeed_raw_segment_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fast.csv"
    write_raw_trajectory(path, second=(2123, 2066, 2058, 2076, 2066))
    with pytest.raises(ValueError, match="speed.*exceeds 5"):
        load_fixed_raw_trajectory(
            path,
            trajectory_id="operator_supplied",
            condition="D2",
            safety=load_safety_envelope(SAFETY_PATH),
        )


def test_alignment_is_bounded_and_does_not_move_frozen_coordinate() -> None:
    safety = load_safety_envelope(SAFETY_PATH)
    start = (2020, 2066, 2058, 2076, 2066)
    target = (2023, 2066, 2055, 2077, 2066)
    events = build_alignment_events(start, target, safety, 5.0)
    assert events[-1].targets_raw == target
    assert all(event.targets_raw[1] == 2066 for event in events)
    previous = start
    previous_time = 0.0
    for event in events:
        required = minimum_safe_command_interval_s(previous, event.targets_raw, safety, 5.0)
        assert event.time_s - previous_time >= required - 1e-12
        previous, previous_time = event.targets_raw, event.time_s


def test_default_cli_is_hardware_free_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "raw.csv"
    write_raw_trajectory(path)
    assert main([
        "--waypoints", str(path),
        "--trajectory-id", "operator_supplied",
        "--condition", "D2",
        "--camera-settings", str(CAMERA_SETTINGS),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_VALIDATED_NO_HARDWARE_ACCESSED"
    assert payload["locked_joint"] == "j2"
    assert "result" not in payload


def test_hardware_runner_dry_run_encodes_every_multi_lock_axis(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "multi.csv"
    write_raw_trajectory(path, condition="J1+J5", second=(2023, 2070, 2058, 2076, 2066))
    assert main([
        "--waypoints", str(path),
        "--trajectory-id", "operator_supplied",
        "--condition", "J1+J5",
        "--camera-settings", str(CAMERA_SETTINGS),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["locked_joints"] == ["j1", "j5"]
    assert payload["locked_targets_raw"] == {"j1": 2023, "j5": 2066}


def test_emergency_torque_off_attempts_all_six_ids_even_after_error() -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.calls = []
            self.state = {servo_id: 1 for servo_id in range(1, 7)}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.calls.append((servo_id, address, value))
            if servo_id == 2:
                raise OSError("injected failure")
            self.state[servo_id] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            return self.state[servo_id]

    bus = FakeBus()
    report = torque_off_all(bus, attempts=3, retry_delay_s=0.0)
    assert [call[0] for call in bus.calls[:6]] == [1, 2, 3, 4, 5, 6]
    assert all(call[2] == 0 for call in bus.calls)
    assert report["status"] == "NOT_VERIFIED_OFF"
    assert report["enabled_ids"] == [2]
    assert len(report["servos"]["2"]["write_attempts"]) == 3


def test_emergency_torque_off_recovers_from_silent_dropped_first_writes() -> None:
    class SilentDropBus:
        def __init__(self) -> None:
            self.state = {servo_id: 1 for servo_id in range(1, 7)}
            self.write_count = {servo_id: 0 for servo_id in range(1, 7)}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.write_count[servo_id] += 1
            if self.write_count[servo_id] > 1:
                self.state[servo_id] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            return self.state[servo_id]

    bus = SilentDropBus()
    report = torque_off_all(bus, attempts=3, retry_delay_s=0.0)
    assert report["status"] == "VERIFIED_OFF"
    assert report["verified_ids"] == [1, 2, 3, 4, 5, 6]
    assert bus.write_count == {servo_id: 2 for servo_id in range(1, 7)}
    assert all(len(report["servos"][str(servo_id)]["read_attempts"]) == 2
               for servo_id in range(1, 7))


def test_emergency_torque_off_retries_after_transient_read_timeout() -> None:
    class TransientReadTimeoutBus:
        def __init__(self) -> None:
            self.state = {servo_id: 1 for servo_id in range(1, 7)}
            self.read_count = {servo_id: 0 for servo_id in range(1, 7)}
            self.write_count = {servo_id: 0 for servo_id in range(1, 7)}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.write_count[servo_id] += 1
            self.state[servo_id] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            self.read_count[servo_id] += 1
            if self.read_count[servo_id] == 1:
                raise TimeoutError("injected transient read timeout")
            return self.state[servo_id]

    bus = TransientReadTimeoutBus()
    report = torque_off_all(bus, attempts=3, retry_delay_s=0.0)
    assert report["status"] == "VERIFIED_OFF"
    assert bus.write_count == {servo_id: 2 for servo_id in range(1, 7)}
    for servo_id in range(1, 7):
        reads = report["servos"][str(servo_id)]["read_attempts"]
        assert "TimeoutError" in reads[0]["error"]
        assert reads[1]["value"] == 0


def test_emergency_torque_off_persists_uncertain_readback_in_report() -> None:
    class UnreadableBus:
        def __init__(self) -> None:
            self.write_count = {servo_id: 0 for servo_id in range(1, 7)}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.write_count[servo_id] += 1

        def read_u8(self, servo_id: int, address: int) -> int:
            raise TimeoutError("persistent timeout")

    bus = UnreadableBus()
    report = torque_off_all(bus, attempts=3, retry_delay_s=0.0)
    assert report["status"] == "NOT_VERIFIED_OFF"
    assert report["uncertain_ids"] == [1, 2, 3, 4, 5, 6]
    assert report["enabled_ids"] == []
    assert bus.write_count == {servo_id: 3 for servo_id in range(1, 7)}
    assert all(len(report["servos"][str(servo_id)]["write_attempts"]) == 3
               for servo_id in range(1, 7))
    manifest = {}
    reports = []
    record_torque_shutdown_report(
        manifest, reports, phase="finally_repeat", report=report
    )
    assert manifest["torque_shutdown_status"] == "NOT_VERIFIED_OFF"
    assert manifest["torque_shutdown_readback_uncertain"] is True
    assert manifest["torque_shutdown_attempts"][0]["uncertain_ids"] == [1, 2, 3, 4, 5, 6]


def test_critical_register_writes_retry_silent_drop_and_read_timeout() -> None:
    class CriticalWriteBus:
        def __init__(self) -> None:
            self.byte = 0
            self.word = 100
            self.byte_writes = 0
            self.word_writes = 0
            self.word_reads = 0

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.byte_writes += 1
            if self.byte_writes > 1:
                self.byte = value

        def read_u8(self, servo_id: int, address: int) -> int:
            return self.byte

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.word_writes += 1
            self.word = value

        def read_u16(self, servo_id: int, address: int) -> int:
            self.word_reads += 1
            if self.word_reads == 1:
                raise TimeoutError("transient target read timeout")
            return self.word

    bus = CriticalWriteBus()
    byte_history = write_u8_verified(
        bus, 1, 40, 1, attempts=3, retry_delay_s=0.0
    )
    word_history = write_u16_verified(
        bus, 1, 42, 2023, attempts=3, retry_delay_s=0.0
    )
    assert bus.byte == 1 and bus.byte_writes == 2
    assert len(byte_history) == 2 and byte_history[0]["readback"] == 0
    assert bus.word == 2023 and bus.word_writes == 2
    assert "TimeoutError" in word_history[0]["read_error"]
    assert word_history[1]["readback"] == 2023


def test_critical_write_failure_is_bounded() -> None:
    class AlwaysDropBus:
        def __init__(self) -> None:
            self.writes = 0

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes += 1

        def read_u16(self, servo_id: int, address: int) -> int:
            return 100

    bus = AlwaysDropBus()
    with pytest.raises(RuntimeError, match="verified u16 write failed"):
        write_u16_verified(bus, 1, 42, 2023, attempts=3, retry_delay_s=0.0)
    assert bus.writes == 3


def test_high_frequency_microstep_writes_changed_axes_without_readback() -> None:
    class WriteOnlyBus:
        def __init__(self) -> None:
            self.writes = []

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes.append((servo_id, address, value))

        def read_u16(self, servo_id: int, address: int) -> int:
            raise AssertionError("microstep must not perform per-dispatch readback")

    bus = WriteOnlyBus()
    changed = command_changed_trajectory_targets(
        bus,
        (2023, 2066, 2058, 2076, 2066),
        (2024, 2066, 2057, 2076, 2066),
    )
    assert changed == (1, 3)
    assert bus.writes == [(1, 42, 2024), (3, 42, 2057)]


def test_goal_validation_schedule_is_bounded_and_final_is_unconditional() -> None:
    schedule = GoalValidationSchedule(
        every_dispatches=10, period_s=0.5, last_validation_s=0.0
    )
    for _ in range(9):
        schedule.note_dispatch()
    assert schedule.due_reason(0.1) is None
    schedule.note_dispatch()
    assert schedule.due_reason(0.1) == "dispatch_count"
    schedule.mark_validated(0.1)
    assert schedule.due_reason(0.59) is None
    assert schedule.due_reason(0.61) == "time_period"
    schedule.mark_validated(0.61)
    assert schedule.final_reason() == "final_event"
    assert schedule.final_reason() == "final_event"


def test_periodic_goal_readback_retries_timeout_and_checks_all_axes() -> None:
    expected = (2023, 2066, 2058, 2076, 2066)

    class TransientGoalReadBus:
        def __init__(self) -> None:
            self.reads = {servo_id: 0 for servo_id in range(1, 6)}

        def read_u16(self, servo_id: int, address: int) -> int:
            self.reads[servo_id] += 1
            if servo_id == 2 and self.reads[servo_id] == 1:
                raise TimeoutError("transient goal timeout")
            return expected[servo_id - 1]

    bus = TransientGoalReadBus()
    event = validate_trajectory_goal_readback(
        bus,
        expected,
        reason=GoalValidationSchedule.final_reason(),
        dispatch_count=10,
        attempts=2,
        retry_delay_s=0.0,
    )
    assert event["status"] == "PASS"
    assert event["reason"] == "final_event"
    assert set(event["axes"]) == {"j1", "j2", "j3", "j4", "j5"}
    assert len(event["axes"]["j2"]["read_attempts"]) == 2


def test_single_silent_microstep_drop_is_corrected_and_preserves_history() -> None:
    initial = (2023, 2066, 2058, 2076, 2066)
    expected = (2024, 2066, 2058, 2076, 2066)

    class SilentDropGoalBus:
        def __init__(self) -> None:
            self.goal = list(initial)
            self.writes = []

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes.append((servo_id, address, value))
            if len(self.writes) > 1:
                self.goal[servo_id - 1] = value
            # Deliberately drop only the first packet without raising.

        def read_u16(self, servo_id: int, address: int) -> int:
            return self.goal[servo_id - 1]

    bus = SilentDropGoalBus()
    assert command_changed_trajectory_targets(bus, initial, expected) == (1,)
    event = validate_trajectory_goal_readback(
        bus,
        expected,
        reason="dispatch_count",
        dispatch_count=10,
        attempts=2,
        retry_delay_s=0.0,
        correction_settle_s=0.0,
    )
    assert event["status"] == "CORRECTED_AFTER_RETRY"
    assert event["expected_targets_raw"] == list(expected)
    assert event["original_problem_axes"] == ["j1"]
    assert event["axes"]["j1"]["status"] == "MISMATCH"
    assert event["axes"]["j1"]["observed_raw"] == initial[0]
    assert event["correction_rounds_used"] == 1
    assert event["correction_history"][0]["targeted_axes"] == ["j1"]
    assert event["correction_history"][0]["unresolved_axes"] == []
    assert event["correction_history"][0]["readback_axes"]["j1"]["status"] == "MATCH"
    assert "last_correction_dispatch_monotonic_ns" in event
    assert bus.writes == [(1, 42, expected[0]), (1, 42, expected[0])]


def test_persistent_silent_microstep_drop_aborts_after_three_corrections() -> None:
    initial = (2023, 2066, 2058, 2076, 2066)
    expected = (2024, 2066, 2058, 2076, 2066)

    class AlwaysDropGoalBus:
        def __init__(self) -> None:
            self.goal = list(initial)
            self.writes = []

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes.append((servo_id, address, value))

        def read_u16(self, servo_id: int, address: int) -> int:
            return self.goal[servo_id - 1]

    bus = AlwaysDropGoalBus()
    assert command_changed_trajectory_targets(bus, initial, expected) == (1,)
    event = validate_trajectory_goal_readback(
        bus,
        expected,
        reason="dispatch_count",
        dispatch_count=10,
        attempts=2,
        retry_delay_s=0.0,
        correction_rounds=3,
        correction_settle_s=0.0,
    )
    assert event["status"] == "FAIL"
    assert event["failure_code"] == "goal_readback_mismatch"
    assert event["failure_joint"] == "j1"
    assert event["axes"]["j1"]["observed_raw"] == initial[0]
    assert len(event["correction_history"]) == 3
    assert all(item["targeted_axes"] == ["j1"]
               for item in event["correction_history"])
    assert bus.writes == [(1, 42, expected[0])] * 4


def test_final_goal_validation_can_correct_confirmed_mismatch() -> None:
    expected = (2023, 2066, 2058, 2076, 2075)

    class FinalCorrectionBus:
        def __init__(self) -> None:
            self.goal = [*expected[:-1], expected[-1] + 1]
            self.writes = []

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes.append((servo_id, address, value))
            self.goal[servo_id - 1] = value

        def read_u16(self, servo_id: int, address: int) -> int:
            return self.goal[servo_id - 1]

    bus = FinalCorrectionBus()
    event = validate_trajectory_goal_readback(
        bus,
        expected,
        reason=GoalValidationSchedule.final_reason(),
        dispatch_count=37,
        retry_delay_s=0.0,
        correction_settle_s=0.0,
    )
    assert event["reason"] == "final_event"
    assert event["status"] == "CORRECTED_AFTER_RETRY"
    assert event["original_problem_axes"] == ["j5"]
    assert event["axes"]["j5"]["observed_raw"] == expected[-1] + 1
    assert event["correction_history"][0]["targeted_axes"] == ["j5"]
    assert bus.writes == [(5, 42, expected[-1])]


def test_persistent_periodic_goal_read_timeout_fails_closed() -> None:
    class TimeoutBus:
        def __init__(self) -> None:
            self.reads = 0
            self.writes = 0

        def read_u16(self, servo_id: int, address: int) -> int:
            self.reads += 1
            raise TimeoutError("persistent goal timeout")

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.writes += 1

    bus = TimeoutBus()
    event = validate_trajectory_goal_readback(
        bus,
        (2023, 2066, 2058, 2076, 2066),
        reason="time_period",
        dispatch_count=4,
        retry_delay_s=0.0,
        correction_settle_s=0.0,
    )
    assert event["status"] == "FAIL"
    assert event["failure_code"] == "goal_readback_timeout"
    assert event["failure_joint"] == "j1"
    assert set(event["axes"]) == {"j1"}
    assert event["correction_history"] == []
    assert bus.reads == TRAJECTORY_GOAL_READ_ATTEMPTS == 2
    assert bus.writes == 0
    assert TRAJECTORY_GOAL_READ_RETRY_DELAY_S <= 0.01


def test_damage_lock_aligns_before_activation_and_uses_feedback_gate() -> None:
    safety = load_safety_envelope(SAFETY_PATH)
    start = (2023, 2063, 2058, 2076, 2066)
    target = (2023, 2066, 2058, 2076, 2066)
    events = build_alignment_events(start, target, safety, 5.0)
    assert [event.targets_raw[1] for event in events] == [2064, 2065, 2066]
    passed = build_damage_activation_record(
        condition="D2",
        locked_index=1,
        target_raw=2066,
        feedback_raw=2064,
        feedback_read_attempts=[{"attempt": 1, "value": 2064}],
        maximum_drift_deg=3.5,
    )
    assert passed["status"] == "PASS_DAMAGE_ACTIVE"
    assert passed["activation_phase"] == "after_start_alignment_before_fixed_trajectory"
    assert passed["feedback_error_ticks"] == 2
    failed = build_damage_activation_record(
        condition="D2",
        locked_index=1,
        target_raw=2066,
        feedback_raw=2025,
        feedback_read_attempts=[{"attempt": 1, "value": 2025}],
        maximum_drift_deg=3.5,
    )
    assert failed["status"] == "FAIL"
    assert failed["failure_code"] == "alignment_lock_error_exceeded"


def test_static_seed_uses_each_present_target_before_torque_on() -> None:
    initial = (2023, 2063, 2058, 2076, 2066)

    class RegisterBus:
        def __init__(self) -> None:
            self.u8 = {(servo_id, 33): 0 for servo_id in range(1, 6)}
            self.u8.update({(servo_id, 40): 0 for servo_id in range(1, 6)})
            self.u16 = {(servo_id, 42): 0 for servo_id in range(1, 6)}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.u8[(servo_id, address)] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            return self.u8[(servo_id, address)]

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.u16[(servo_id, address)] = value

        def read_u16(self, servo_id: int, address: int) -> int:
            return self.u16[(servo_id, address)]

    bus = RegisterBus()
    seed_report, torque_report = seed_and_enable_servos(
        bus, initial, settle_s=0.0
    )
    assert seed_report["status"] == "PASS"
    assert torque_report["status"] == "PASS"
    assert tuple(bus.u16[(servo_id, 42)] for servo_id in range(1, 6)) == initial
    assert all(bus.u8[(servo_id, 40)] == 1 for servo_id in range(1, 6))


def test_static_batch_writes_every_axis_then_settles_before_any_read() -> None:
    targets = (2023, 2063, 2058, 2076, 2066)
    events: list[tuple[object, ...]] = []

    class SettleSensitiveBus:
        def __init__(self) -> None:
            self.settled = False
            self.u8: dict[tuple[int, int], int] = {}
            self.u16: dict[tuple[int, int], int] = {}

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            events.append(("write_u8", servo_id, address, value))
            self.settled = False
            self.u8[(servo_id, address)] = value

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            events.append(("write_u16", servo_id, address, value))
            self.settled = False
            self.u16[(servo_id, address)] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            events.append(("read_u8", servo_id, address))
            if not self.settled:
                raise TimeoutError("device is not ready immediately after write")
            return self.u8[(servo_id, address)]

        def read_u16(self, servo_id: int, address: int) -> int:
            events.append(("read_u16", servo_id, address))
            if not self.settled:
                raise TimeoutError("device is not ready immediately after write")
            return self.u16[(servo_id, address)]

    bus = SettleSensitiveBus()

    def settle(duration_s: float) -> None:
        events.append(("sleep", duration_s))
        assert duration_s == STATIC_BATCH_SETTLE_S
        bus.settled = True

    report = batch_write_and_verify_static_registers(
        bus,
        torque_off_static_configuration_expectations(targets),
        phase="torque_off_goal_accel_speed_before_cameras",
        sleep_fn=settle,
    )
    assert report["status"] == "PASS"
    assert len(report["initial_writes"]) == 15
    assert len(report["initial_readbacks"]) == 15
    first_read = next(index for index, event in enumerate(events)
                      if str(event[0]).startswith("read_"))
    settle_index = events.index(("sleep", STATIC_BATCH_SETTLE_S))
    assert settle_index == 15
    assert first_read == 16
    assert all(str(event[0]).startswith("write_") for event in events[:15])


def test_static_batch_only_rewrites_confirmed_mismatch() -> None:
    expectations = (
        StaticRegisterExpectation("j1.acceleration", 1, 41, 8, 1),
        StaticRegisterExpectation("j2.acceleration", 2, 41, 8, 1),
        StaticRegisterExpectation("j3.acceleration", 3, 41, 8, 1),
    )

    class OneSilentDropBus:
        def __init__(self) -> None:
            self.values = {(1, 41): 0, (2, 41): 0, (3, 41): 0}
            self.writes: list[tuple[int, int, int]] = []

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.writes.append((servo_id, address, value))
            if servo_id != 2 or sum(call[0] == 2 for call in self.writes) > 1:
                self.values[(servo_id, address)] = value

        def read_u8(self, servo_id: int, address: int) -> int:
            return self.values[(servo_id, address)]

    bus = OneSilentDropBus()
    report = batch_write_and_verify_static_registers(
        bus,
        expectations,
        phase="test_static_mismatch",
        settle_s=0.0,
        read_retry_delay_s=0.0,
    )
    assert report["status"] == "CORRECTED_AFTER_RETRY"
    assert report["original_mismatch_labels"] == ["j2.acceleration"]
    assert report["correction_history"][0]["targeted_labels"] == [
        "j2.acceleration"
    ]
    assert bus.writes == [
        (1, 41, 1), (2, 41, 1), (3, 41, 1), (2, 41, 1)
    ]


def test_static_batch_persistent_read_timeout_fails_without_rewrite() -> None:
    expectation = StaticRegisterExpectation("j4.acceleration", 4, 41, 8, 1)

    class UnreadableAfterWriteBus:
        def __init__(self) -> None:
            self.writes = 0
            self.reads = 0

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.writes += 1

        def read_u8(self, servo_id: int, address: int) -> int:
            self.reads += 1
            raise TimeoutError("persistent static read timeout")

    bus = UnreadableAfterWriteBus()
    report = batch_write_and_verify_static_registers(
        bus,
        (expectation,),
        phase="test_static_timeout",
        settle_s=0.0,
        read_retry_delay_s=0.0,
    )
    assert report["status"] == "FAIL"
    assert report["failure_code"] == "static_register_read_timeout"
    assert report["failure_label"] == "j4.acceleration"
    assert report["correction_history"] == []
    assert bus.reads == 2
    assert bus.writes == 1


def test_seed_readback_failure_never_reaches_torque_enable_batch() -> None:
    class SeedUnreadableBus:
        def __init__(self) -> None:
            self.u8_writes: list[tuple[int, int, int]] = []
            self.goal_writes: list[tuple[int, int, int]] = []

        def read_u8(self, servo_id: int, address: int) -> int:
            assert address == 33
            return 0

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            self.u8_writes.append((servo_id, address, value))

        def write_u16(self, servo_id: int, address: int, value: int) -> None:
            self.goal_writes.append((servo_id, address, value))

        def read_u16(self, servo_id: int, address: int) -> int:
            raise TimeoutError("seed goal readback unavailable")

    bus = SeedUnreadableBus()
    with pytest.raises(RuntimeError, match="latest_present_goal_seed"):
        seed_and_enable_servos(
            bus,
            (2023, 2063, 2058, 2076, 2066),
            settle_s=0.0,
            sleep_fn=lambda _seconds: None,
        )
    assert len(bus.goal_writes) == 5
    assert bus.u8_writes == []


def test_twenty_fps_deadlines_produce_twenty_frames_per_second() -> None:
    starts = [0]
    for _ in range(19):
        starts.append(_next_capture_deadline_ns(starts[-1], 20.0))
    assert len(starts) == 20
    assert starts[-1] == 950_000_000
    assert _next_capture_deadline_ns(starts[-1], 20.0) == 1_000_000_000
    summary = validate_recorder_cadence(
        frame_count=20,
        first_capture_mid_ns=starts[0],
        last_capture_mid_ns=starts[-1],
        nominal_fps=20.0,
    )
    assert summary["observed_fps"] == pytest.approx(20.0)


def test_nominal_twenty_fps_rejects_unthrottled_168_fps_timestamps() -> None:
    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_recorder_cadence(
            frame_count=169,
            first_capture_mid_ns=0,
            last_capture_mid_ns=1_000_000_000,
            nominal_fps=20.0,
        )


def test_telemetry_schema_has_measured_and_commanded_columns_for_all_joints() -> None:
    fields = telemetry_fields()
    for joint in ("j1", "j2", "j3", "j4", "j5"):
        assert f"{joint}_position_raw" in fields
        assert f"{joint}_target_raw" in fields
        assert f"{joint}_voltage_v" in fields
        assert f"{joint}_temperature_c" in fields
        assert f"{joint}_current_raw" in fields


def _run_mocked_hardware_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_camera_start: bool = False,
    powered_handoff: bool = False,
) -> tuple[list[tuple[object, ...]], Path]:
    events: list[tuple[object, ...]] = []
    initial = (2023, 2066, 2058, 2076, 2066)
    latest = (2024, 2066, 2058, 2076, 2066)

    class FakeBus:
        READ_DEADLINE_S = 0.12

        def __init__(self, _port: str) -> None:
            self.serial = SimpleNamespace(timeout=0.08)
            events.append(("bus_open",))

        def write_u8(self, servo_id: int, address: int, value: int) -> None:
            events.append(("write_u8", servo_id, address, value))

        def read_u8(self, servo_id: int, address: int) -> int:
            events.append(("read_u8", servo_id, address))
            if powered_handoff and address == runner.ADDRESS_TORQUE_ENABLE:
                return 1
            return 0

        def close(self) -> None:
            events.append(("bus_close",))

    class FakeRecorder:
        def __init__(self, output: Path, *_args: object) -> None:
            self.output = output
            self.frame_count = 1
            self.first_capture_mid_ns = 0
            self.last_capture_mid_ns = 0

        def start(self) -> None:
            events.append(("camera_start", self.output.name))
            if fail_camera_start:
                raise RuntimeError("injected camera start failure")

        def wait_ready(self, _timeout_s: float) -> bool:
            events.append(("camera_ready", self.output.name))
            return True

        def stop(self) -> None:
            events.append(("camera_stop", self.output.name))

    import scripts.recover_j5 as recover_j5

    monkeypatch.setattr(recover_j5, "ServoBus", FakeBus)
    monkeypatch.setattr(runner, "DirectShowRecorder", FakeRecorder)
    monkeypatch.setattr(runner, "DahengRecorder", FakeRecorder)
    original_torque_off = runner.torque_off_all

    def logged_torque_off(bus: object) -> dict[str, object]:
        events.append(("torque_off",))
        return original_torque_off(bus, attempts=1, retry_delay_s=0.0)

    monkeypatch.setattr(runner, "torque_off_all", logged_torque_off)
    positions = iter((initial, latest) if powered_handoff else (initial, initial, latest))

    def read_positions(_bus: object) -> tuple[int, int, int, int, int]:
        value = next(positions)
        events.append(("read_positions", value))
        return value

    monkeypatch.setattr(runner, "read_positions", read_positions)

    def pass_batch(
        _bus: object, expectations: object, *, phase: str
    ) -> dict[str, object]:
        expected = tuple(item.expected for item in expectations)  # type: ignore[attr-defined]
        events.append(("batch", phase, expected))
        return {"phase": phase, "status": "PASS"}

    monkeypatch.setattr(runner, "batch_write_and_verify_static_registers", pass_batch)
    monkeypatch.setattr(runner, "build_alignment_events", lambda *_args: ())
    monkeypatch.setattr(runner, "interpolate_raw_waypoints", lambda _trajectory: ())
    monkeypatch.setattr(runner, "read_and_validate_telemetry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "validate_trajectory_goal_readback",
        lambda *_args, **_kwargs: {"status": "PASS", "axes": {}},
    )
    monkeypatch.setattr(runner, "validate_recorder_cadence", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "validate_video_file", lambda *_args, **_kwargs: {})

    safety = load_safety_envelope(SAFETY_PATH)
    trajectory = SimpleNamespace(
        condition="intact",
        locked_joint_index=None,
        waypoints=(SimpleNamespace(targets_raw=initial),),
    )
    args = SimpleNamespace(
        acknowledge_risk=runner.ACKNOWLEDGEMENT,
        trial_id="mock_failure" if fail_camera_start else "mock_success",
        telemetry_hz=20.0,
        video_fps=20.0,
        camera_ready_timeout_s=1.0,
        pre_roll_s=0.0,
        post_roll_s=0.0,
        maximum_start_error_deg=2.0,
        minimum_voltage_v=6.0,
        safety=SAFETY_PATH,
        output_root=tmp_path,
        port="MOCK",
        sdk_root=tmp_path,
        maximum_speed_deg_s=5.0,
        startup_powered_handoff=powered_handoff,
        keep_torque_enabled_after_success=powered_handoff,
    )
    trial_dir = tmp_path / args.trial_id
    if fail_camera_start:
        with pytest.raises(RuntimeError, match="injected camera start failure"):
            runner.execute_hardware(args, {"camera_settings": {}}, trajectory, safety)
    else:
        assert runner.execute_hardware(
            args, {"camera_settings": {}}, trajectory, safety
        ) == trial_dir.resolve()
    return events, trial_dir


def test_execute_hardware_orders_startup_seed_enable_and_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, trial_dir = _run_mocked_hardware_executor(tmp_path, monkeypatch)
    first_torque = events.index(("torque_off",))
    assert events[first_torque + 1:first_torque + 7] == [
        ("write_u8", servo_id, runner.ADDRESS_TORQUE_ENABLE, 0)
        for servo_id in range(1, 7)
    ]
    static = next(i for i, item in enumerate(events)
                  if item[:2] == ("batch", "torque_off_goal_accel_speed_before_cameras"))
    camera = next(i for i, item in enumerate(events) if item[0] == "camera_start")
    latest = events.index(("read_positions", (2024, 2066, 2058, 2076, 2066)))
    seed = next(i for i, item in enumerate(events)
                if item[:2] == ("batch", "latest_present_goal_seed"))
    enable = next(i for i, item in enumerate(events)
                  if item[:2] == ("batch", "torque_enable_last"))
    assert first_torque < static < camera < latest < seed < enable
    assert events[seed][2] == (2024, 2066, 2058, 2076, 2066)
    assert sum(item == ("torque_off",) for item in events) == 3
    manifest = json.loads((trial_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["servo_single_read_deadline_s"] == pytest.approx(0.12)
    assert manifest["runtime_goal_single_axis_timeout_bound_s"] == pytest.approx(0.249)
    assert manifest["runtime_goal_single_axis_timeout_bound_s"] * 1000 < 250


def test_execute_hardware_exception_requests_two_independent_shutdowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, trial_dir = _run_mocked_hardware_executor(
        tmp_path, monkeypatch, fail_camera_start=True
    )
    assert sum(item == ("torque_off",) for item in events) == 3
    manifest = json.loads((trial_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert [item["phase"] for item in manifest["torque_shutdown_attempts"]] == [
        "exception_immediate", "finally_repeat"
    ]


def test_powered_handoff_success_never_drops_torque_and_leaves_hold_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, trial_dir = _run_mocked_hardware_executor(
        tmp_path, monkeypatch, powered_handoff=True
    )
    assert ("torque_off",) not in events
    assert not any(item[:2] == ("batch", "torque_enable_last") for item in events)
    manifest = json.loads((trial_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["startup_torque_enable_readback_ids_1_5"] == [1] * 5
    assert manifest["completion_torque_enable_readback_ids_1_5"] == [1] * 5
    assert manifest["powered_hold_active"] is True
    assert manifest["status"] == "ACQUISITION_COMPLETE_UNASSESSED"


def test_powered_handoff_exception_still_requests_two_shutdowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, trial_dir = _run_mocked_hardware_executor(
        tmp_path, monkeypatch, fail_camera_start=True, powered_handoff=True
    )
    assert sum(item == ("torque_off",) for item in events) == 2
    manifest = json.loads((trial_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert [item["phase"] for item in manifest["torque_shutdown_attempts"]] == [
        "exception_immediate", "finally_repeat"
    ]
    assert manifest["powered_hold_active"] is False


def test_servo_read_clamps_each_blocking_read_to_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class FakeSerial:
        def __init__(self) -> None:
            self.timeout = 0.08
            self.read_timeouts: list[float] = []

        def reset_input_buffer(self) -> None:
            pass

        def write(self, packet: bytes) -> int:
            return len(packet)

        def flush(self) -> None:
            raise AssertionError("read request must not use unbounded flush")

        def read(self, _size: int) -> bytes:
            self.read_timeouts.append(self.timeout)
            now[0] += self.timeout
            return b""

    monkeypatch.setattr("scripts.recover_j5.time.monotonic", lambda: now[0])
    bus = object.__new__(RecoverServoBus)
    bus.serial = FakeSerial()
    with pytest.raises(TimeoutError, match="did not respond"):
        bus.read(1, 42, 2)
    assert bus.serial.read_timeouts == pytest.approx([0.08, 0.04])
    assert now[0] == pytest.approx(bus.READ_DEADLINE_S)
    assert bus.serial.timeout == pytest.approx(0.08)


def test_teach_capture_is_read_only_and_output_loads(tmp_path: Path) -> None:
    class ReadOnlyFakeBus:
        def __init__(self) -> None:
            self.calls = []

        def read_u16(self, servo_id: int, address: int) -> int:
            self.calls.append((servo_id, address))
            return (2023, 2066, 2058, 2076, 2066)[servo_id - 1]

    safety = load_safety_envelope(SAFETY_PATH)
    bus = ReadOnlyFakeBus()
    raw = capture_positions_readonly(bus)
    assert bus.calls == [(1, 56), (2, 56), (3, 56), (4, 56), (5, 56)]
    assert raw_to_radians(raw, safety) == pytest.approx((0.0,) * 5)
    output = tmp_path / "taught.csv"
    append_waypoint(
        output,
        trajectory_id="human_named_d3",
        condition="D3",
        waypoint_index=0,
        time_s=0.0,
        raw=raw,
        safety=safety,
        captured_utc="2026-09-01T00:00:00+00:00",
    )
    moved = (2024, 2065, 2058, 2076, 2066)
    append_waypoint(
        output,
        trajectory_id="human_named_d3",
        condition="D3",
        waypoint_index=1,
        time_s=1.0,
        raw=moved,
        safety=safety,
        captured_utc="2026-09-01T00:00:01+00:00",
    )
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or ()) == FIELDS
        assert len(list(reader)) == 2
    trajectory = load_fixed_raw_trajectory(
        output,
        trajectory_id="human_named_d3",
        condition="D3",
        safety=safety,
    )
    assert trajectory.waypoints[-1].targets_raw == moved
    schedule = tmp_path / "schedule.csv"
    with schedule.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "condition"])
        writer.writeheader()
        writer.writerow({"trajectory_id": "human_named_d3", "condition": "D3"})
    assert audit_level_a_library(output, schedule)["status"] == "PASS"


def test_teach_append_refuses_skipped_index(tmp_path: Path) -> None:
    safety = load_safety_envelope(SAFETY_PATH)
    output = tmp_path / "taught.csv"
    raw = (2023, 2066, 2058, 2076, 2066)
    append_waypoint(
        output,
        trajectory_id="named",
        condition="intact",
        waypoint_index=0,
        time_s=0.0,
        raw=raw,
        safety=safety,
        captured_utc="now",
    )
    with pytest.raises(ValueError, match="must be 1"):
        append_waypoint(
            output,
            trajectory_id="named",
            condition="intact",
            waypoint_index=2,
            time_s=1.0,
            raw=raw,
            safety=safety,
            captured_utc="later",
        )
