import json
from pathlib import Path
import subprocess
import sys


def test_feasibility_builder_marks_claim_boundary(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "physical_feasibility_claim_level": "formal",
        "physical_feasibility_by_condition": {
            name: {"valid_trials": 10, "aborted_trials": 0,
                   "mean_endpoint_error_m": 0.02, "reach_rate": 1.0,
                   "contact_rate": 0.9, "success_rate": 0.8,
                   "max_lock_error_rad": 0.02}
            for name in ("intact", "D2", "D3")
        },
    }), encoding="utf-8")
    figure, table = tmp_path / "figure.pdf", tmp_path / "table.tex"
    completed = subprocess.run(
        [sys.executable, "scripts/build_real_robot_feasibility_assets.py",
         str(summary), "--figure", str(figure), "--table", str(table)],
        cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert figure.read_bytes().startswith(b"%PDF")
    text = table.read_text(encoding="utf-8")
    assert "D3 & 10 & 0" in text
    assert "does not support learned-method superiority" in text


def test_feasibility_builder_rejects_missing_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "physical_feasibility_claim_level": "no physical evidence"}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/build_real_robot_feasibility_assets.py",
         str(summary), "--figure", str(tmp_path / "x.pdf"),
         "--table", str(tmp_path / "x.tex")], cwd=root,
        capture_output=True, text=True)
    assert completed.returncode != 0
