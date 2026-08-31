import csv
from pathlib import Path

from scripts.audit_level_a_trajectory_library import audit
from scripts.generate_real_robot_level_a_schedule import FIELDS, build


def write_schedule(path: Path) -> None:
    rows = build(1, 10, {"intact": "i", "D2": "d2", "D3": "d3"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def write_library(path: Path, d2_motion: float = 0.0,
                  speed_motion: float = 0.01) -> None:
    fields = ["trajectory_id", "condition", "waypoint_index", "time_s",
              "j1", "j2", "j3", "j4", "j5"]
    rows = []
    for trajectory, condition in (("i", "intact"), ("d2", "D2"), ("d3", "D3")):
        rows.extend([
            dict(zip(fields, [trajectory, condition, 0, 0, 0, 0, 0, 0, 0])),
            dict(zip(fields, [trajectory, condition, 1, 1, speed_motion,
                              d2_motion if condition == "D2" else speed_motion,
                              0 if condition == "D3" else speed_motion,
                              speed_motion, speed_motion])),
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_safe_library_matches_schedule_and_limits(tmp_path: Path) -> None:
    schedule, library = tmp_path / "schedule.csv", tmp_path / "library.csv"
    write_schedule(schedule); write_library(library)
    result = audit(library, schedule)
    assert result["status"] == "PASS"
    assert result["authorization"] == "TRAJECTORY_LIBRARY_SAFE_TO_FREEZE"


def test_locked_joint_motion_is_rejected(tmp_path: Path) -> None:
    schedule, library = tmp_path / "schedule.csv", tmp_path / "library.csv"
    write_schedule(schedule); write_library(library, d2_motion=0.01)
    result = audit(library, schedule)
    assert result["status"] == "FAIL"
    assert any("locked j2" in error for error in result["errors"])


def test_overspeed_motion_is_rejected(tmp_path: Path) -> None:
    schedule, library = tmp_path / "schedule.csv", tmp_path / "library.csv"
    write_schedule(schedule); write_library(library, speed_motion=0.2)
    result = audit(library, schedule)
    assert result["status"] == "FAIL"
    assert any("exceeds 5" in error for error in result["errors"])
