"""Pure-computation tests for the provisional candidate FK audit."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.audit_real_push_candidate_fk import (
    AUTHORIZATION,
    DEFAULT_CANDIDATES,
    MODEL_STATUS,
    audit_candidate_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_three_candidates_are_fully_audited_offline() -> None:
    payload = audit_candidate_file(DEFAULT_CANDIDATES)

    assert payload["status"] == "PASS"
    assert payload["model_status"] == MODEL_STATUS == "provisional"
    assert payload["authorization"] == (
        "LOW_SPEED_PILOT_CANDIDATE_NOT_FORMAL_TRAJECTORY"
    ) == AUTHORIZATION
    assert payload["trajectory_count"] == 3
    assert len(payload["candidate_file_sha256"]) == 64
    assert len(payload["safety_file_sha256"]) == 64
    assert payload["interpretation"]["hardware_resources_accessed"] is False

    conditions = {
        item["condition"] for item in payload["trajectories"].values()
    }
    assert conditions == {"intact", "D2", "D3"}
    for item in payload["trajectories"].values():
        assert item["parser_status"] == "PASS"
        assert item["interpolated_event_count"] > item["waypoint_count"]
        assert item["all_events_within_joint_limits"] is True
        assert item["maximum_interpolated_speed_deg_s"] <= 5.0 + 1e-8
        assert item["tcp"]["start_to_end_dx_m"] > 0.029
        assert item["tcp"]["lateral_y_range_m"] >= 0.0
        assert item["tcp"]["vertical_z_range_m"] >= 0.0

    assert payload["trajectories"]["pilot_intact_fk30mm_v1"][
        "lock_axis_audit"
    ]["applicable"] is False
    for trajectory_id, joint in (
        ("pilot_D2_fk30mm_v1", "j2"),
        ("pilot_D3_fk30mm_v1", "j3"),
    ):
        lock = payload["trajectories"][trajectory_id]["lock_axis_audit"]
        assert lock["joint"] == joint
        assert lock["command_exactly_constant_for_all_events"] is True
        assert lock["raw_range_ticks"] == 0


def test_invalid_locked_axis_is_rejected_by_formal_parser(tmp_path: Path) -> None:
    source_rows: list[dict[str, str]]
    with DEFAULT_CANDIDATES.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        source_rows = list(reader)
    for row in source_rows:
        if row["trajectory_id"] == "pilot_D2_fk30mm_v1" and row["waypoint_index"] == "1":
            row["j2_raw"] = str(int(row["j2_raw"]) + 1)
            # Keep radians consistent so rejection is specifically the lock invariant.
            row["j2"] = "0.877437010671"
            break
    candidate = tmp_path / "bad_lock.csv"
    with candidate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    payload = audit_candidate_file(candidate)

    assert payload["status"] == "FAIL"
    assert payload["authorization"] == AUTHORIZATION
    failed = payload["trajectories"]["pilot_D2_fk30mm_v1"]
    assert failed["parser_status"] == "FAIL"
    assert "requires locked j2 target" in failed["error"]
