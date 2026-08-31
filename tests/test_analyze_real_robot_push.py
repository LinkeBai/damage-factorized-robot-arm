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
    assert payload["paired_relative_endpoint_error_reduction"] == pytest.approx(0.6)
    assert payload["paired_failure_analysis"]["relative_failure_rate_reduction"] == 1.0
    assert payload["paired_failure_analysis"]["discordant_success_pairs"] == {
        "candidate_rescues_reference_failure": 1,
        "candidate_breaks_reference_success": 0,
    }
    assert payload["paired_comparison"]["pairs_by_condition"] == {"D3": 1}
    assert payload["paired_by_condition"]["D3"]["pairs"] == 1
    assert payload["paired_rows"][0]["pair_id"] == "P001"
    assert payload["claim_level"] == "pilot"
    assert payload["formal_gate"]["counts_met"] is False
    assert payload["physical_feasibility_by_condition"]["D3"]["valid_trials"] == 2
    assert payload["physical_feasibility_by_condition"]["D3"]["contact_rate"] == 1.0
    assert payload["physical_feasibility_claim_level"] == "pilot"


def test_failure_reduction_is_none_when_reference_has_no_failures(tmp_path):
    source, output = tmp_path / "trials.csv", tmp_path / "summary.json"
    write_rows(source, [row("nominal", "1", "0.02", "1"),
                        row("global_matched", "2", "0.01", "1")])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_push.py", str(source),
         "--output", str(output)], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["paired_failure_analysis"]["relative_failure_rate_reduction"] is None


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


def test_analyzer_reports_aborted_method_as_incomplete_pair(tmp_path):
    source, output = tmp_path / "trials.csv", tmp_path / "summary.json"
    aborted = row("global_matched", "2", "", "")
    aborted.update({"aborted": "1", "failure_code": "safety_stop"})
    write_rows(source, [row("nominal", "1", "0.05", "0"), aborted])
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_real_robot_push.py", str(source),
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["paired_trials"] == 0
    assert payload["claim_level"] == "no paired evidence"
    incomplete = payload["paired_comparison"]["incomplete_or_aborted_pairs"]
    assert incomplete == [{
        "condition": "D3", "pair_id": "P001",
        "present_methods": ["global_matched", "nominal"],
        "aborted_methods": ["global_matched"],
    }]
