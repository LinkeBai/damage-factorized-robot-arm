import json
from pathlib import Path
import subprocess
import sys


def test_builder_creates_figure_and_table_from_validated_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.json"
    metric = lambda mean: {"mean": mean, "ci95": [mean - 0.001, mean + 0.001]}
    summary.write_text(json.dumps({
        "claim_level": "pilot",
        "all_required_files_checked": False,
        "paired_comparison": {
            "reference_method": "nominal", "candidate_method": "global_matched"
        },
        "paired_by_condition": {
            "D3": {
                "pairs": 2,
                "endpoint_improvement_m": metric(0.003),
                "success_improvement": metric(0.5),
                "relative_failure_rate_reduction": 0.5,
                "reach_improvement": metric(0.0),
                "contact_improvement": metric(0.5),
            }
        },
    }), encoding="utf-8")
    figure, table = tmp_path / "figure.pdf", tmp_path / "table.tex"
    completed = subprocess.run(
        [sys.executable, "scripts/build_real_robot_paper_assets.py", str(summary),
         "--figure", str(figure), "--table", str(table)],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert figure.read_bytes().startswith(b"%PDF")
    text = table.read_text(encoding="utf-8")
    assert "D3 & 2 & 3.00" in text
    assert "50.0" in text
    assert "Positive values favor global_matched over nominal" in text


def test_builder_refuses_empty_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"claim_level": "no paired evidence"}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/build_real_robot_paper_assets.py", str(summary),
         "--figure", str(tmp_path / "x.pdf"), "--table", str(tmp_path / "x.tex")],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "without paired evidence" in (completed.stdout + completed.stderr)
