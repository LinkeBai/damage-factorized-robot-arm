import csv
import json
from pathlib import Path
import subprocess
import sys


FIELDS = [
    "trial_id", "condition", "method", "position_id", "trial_order", "aborted",
    "max_lock_error_rad", "pregrasp_reached", "gripper_closed", "lift_height_m",
    "retained_after_3s", "success", "camera_left_video",
    "camera_horizontal_video", "control_log", "failure_code",
]


def valid_row() -> dict[str, str]:
    return {
        "trial_id": "G001", "condition": "D3", "method": "fixed_pregrasp",
        "position_id": "A", "trial_order": "1", "aborted": "0",
        "max_lock_error_rad": "0.01", "pregrasp_reached": "1",
        "gripper_closed": "1", "lift_height_m": "0.04",
        "retained_after_3s": "1", "success": "1",
        "camera_left_video": "left.mp4",
        "camera_horizontal_video": "horizontal.mp4", "control_log": "log.csv",
        "failure_code": "",
    }


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_grasp_analyzer_reports_feasibility_without_method_claim(tmp_path):
    source, output = tmp_path / "grasp.csv", tmp_path / "summary.json"
    write(source, [valid_row()])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_grasp.py", str(source),
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["claim_level"] == "feasibility"
    assert payload["conditions"]["D3"]["retention_rate_3s"] == 1.0
    assert "no learned-grasp claim" in payload["scope"]


def test_grasp_analyzer_preserves_abort_reason(tmp_path):
    source, output = tmp_path / "grasp.csv", tmp_path / "summary.json"
    row = valid_row()
    row.update({"aborted": "1", "failure_code": "safety_stop"})
    write(source, [row])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_grasp.py", str(source),
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["valid_rows"] == 0
    assert payload["aborted_rows"] == 1
    assert payload["failure_codes"] == {"safety_stop": 1}
