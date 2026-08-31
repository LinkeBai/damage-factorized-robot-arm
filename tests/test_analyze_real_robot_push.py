import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


FIELDS = [
    "pair_id", "condition", "method", "position_id", "trial_order", "aborted",
    "max_lock_error_rad", "reached", "contact", "endpoint_error_m", "success",
    "camera_left_video", "camera_horizontal_video", "control_log", "failure_code",
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(method: str, order: str, endpoint: str, success: str, position: str = "A"):
    return {
        "pair_id": "P001", "condition": "D3", "method": method,
        "position_id": position, "trial_order": order, "aborted": "0",
        "max_lock_error_rad": "0.01", "reached": "1", "contact": "1",
        "endpoint_error_m": endpoint, "success": success,
        "camera_left_video": "left.mp4", "camera_horizontal_video": "horizontal.mp4",
        "control_log": "control.csv", "failure_code": "",
    }


def test_analyzer_reports_configurable_paired_comparison(tmp_path):
    source, output = tmp_path / "trials.csv", tmp_path / "summary.json"
    write_rows(source, [row("nominal", "1", "0.05", "0"),
                        row("global_matched", "2", "0.02", "1")])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_push.py", str(source),
         "--reference-method", "nominal", "--candidate-method", "global_matched",
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["paired_trials"] == 1
    assert payload["paired_endpoint_improvement_m"]["mean"] == pytest.approx(0.03)
    assert payload["paired_success_improvement"]["mean"] == 1.0
    assert payload["paired_comparison"]["pairs_by_condition"] == {"D3": 1}


def test_analyzer_rejects_mismatched_pair_positions(tmp_path):
    source = tmp_path / "trials.csv"
    write_rows(source, [row("nominal", "1", "0.05", "0", "A"),
                        row("global_matched", "2", "0.02", "1", "B")])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_push.py", str(source)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "mismatched reset positions" in (completed.stdout + completed.stderr)
