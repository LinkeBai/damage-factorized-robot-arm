import csv
from pathlib import Path

from scripts.audit_real_robot_schedule_completion import audit
from scripts.generate_real_robot_level_a_schedule import FIELDS, build


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def schedules(tmp_path: Path):
    rows = build(1, 10, {"intact": "i1", "D2": "d2", "D3": "d3"})
    frozen, completed = tmp_path / "schedule.csv", tmp_path / "completed.csv"
    write(frozen, rows)
    filled = [dict(row) for row in rows if row["trial_role"] == "primary"]
    for row in filled:
        row.update({"max_lock_error_rad": "0.01", "reached": "1",
                    "contact": "1", "endpoint_error_m": "0.02", "success": "1",
                    "camera_left_video": "l.mp4",
                    "camera_horizontal_video": "h.mp4",
                    "control_log": "c.csv"})
    write(completed, filled)
    return frozen, completed


def test_completed_log_may_fill_measurements_without_changing_identity(tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    result = audit(frozen, completed)
    assert result["status"] == "PASS"
    assert result["frozen_schedule_sha256"] != result["completed_log_sha256"]
    assert result["changed_identity_rows"] == []


def test_changed_trajectory_is_rejected(tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    rows = list(csv.DictReader(completed.open(newline="", encoding="utf-8")))
    rows[0]["trajectory_id"] = "posthoc_better_motion"
    write(completed, rows)
    result = audit(frozen, completed)
    assert result["status"] == "FAIL"
    assert result["changed_identity_rows"][0]["differences"]["trajectory_id"]


def test_deleted_trial_is_rejected(tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    rows = list(csv.DictReader(completed.open(newline="", encoding="utf-8")))
    write(completed, rows[:-1])
    result = audit(frozen, completed)
    assert result["status"] == "FAIL"
    assert any("removed frozen primary" in error for error in result["errors"])


def test_aborted_primary_is_preserved_and_next_frozen_reserve_reaches_target(
        tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    frozen_rows = list(csv.DictReader(frozen.open(newline="", encoding="utf-8")))
    rows = list(csv.DictReader(completed.open(newline="", encoding="utf-8")))
    aborted = next(row for row in rows if row["condition"] == "D2")
    aborted.update({
        "aborted": "1", "max_lock_error_rad": "", "reached": "", "contact": "",
        "endpoint_error_m": "", "success": "", "camera_left_video": "",
        "camera_horizontal_video": "", "control_log": "",
        "failure_code": "camera_loss",
    })
    reserve = dict(next(
        row for row in frozen_rows
        if row["condition"] == "D2" and row["trial_role"] == "reserve"
        and row["reserve_rank"] == "1"
    ))
    reserve.update({
        "max_lock_error_rad": "0.01", "reached": "1", "contact": "1",
        "endpoint_error_m": "0.02", "success": "1",
        "camera_left_video": "l.mp4", "camera_horizontal_video": "h.mp4",
        "control_log": "c.csv",
    })
    rows.append(reserve)
    write(completed, rows)

    result = audit(frozen, completed)

    assert result["status"] == "PASS"
    assert result["reserve_execution_by_condition"]["D2"]["primary_aborts"] == 1
    assert result["reserve_execution_by_condition"]["D2"]["reserves_executed"] == 1
    assert result["reserve_execution_by_condition"]["D2"]["valid_trials"] == 10


def test_posthoc_extra_or_skipped_reserve_is_rejected(tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    frozen_rows = list(csv.DictReader(frozen.open(newline="", encoding="utf-8")))
    rows = list(csv.DictReader(completed.open(newline="", encoding="utf-8")))
    reserve = dict(next(
        row for row in frozen_rows
        if row["condition"] == "D2" and row["trial_role"] == "reserve"
        and row["reserve_rank"] == "2"
    ))
    reserve.update({
        "max_lock_error_rad": "0.01", "reached": "1", "contact": "1",
        "endpoint_error_m": "0.02", "success": "1",
        "camera_left_video": "l.mp4", "camera_horizontal_video": "h.mp4",
        "control_log": "c.csv",
    })
    rows.append(reserve)
    write(completed, rows)

    result = audit(frozen, completed)

    assert result["status"] == "FAIL"
    assert any("required frozen prefix" in error for error in result["errors"])


def test_outcome_and_lock_thresholds_are_audited(tmp_path: Path) -> None:
    frozen, completed = schedules(tmp_path)
    rows = list(csv.DictReader(completed.open(newline="", encoding="utf-8")))
    rows[0]["endpoint_error_m"] = "0.031"
    rows[0]["success"] = "1"
    rows[1]["max_lock_error_rad"] = "0.062"
    write(completed, rows)

    result = audit(frozen, completed)

    assert result["status"] == "FAIL"
    assert result["inconsistent_success_rows"]
    assert result["lock_error_violations"]
