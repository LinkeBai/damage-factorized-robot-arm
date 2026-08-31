import json
import subprocess
import sys
from pathlib import Path


def payload(model_value):
    def metrics(value):
        return {"constraint": {"locked_position_violation_max": 0.0, "locked_velocity_violation_max": 0.0},
                "response": {"contact_candidate_terminal_object_rmse": value},
                "action_ranking": {"top1_regret": value, "spearman": 0.2},
                "closed_loop_outcome": {"endpoint_error": value, "success_rate": 0.3}}
    return {"formal_six_stage_metrics": {"shared_baseline": metrics(2.0),
        "projection_global_residual_matched": metrics(model_value)}}


def test_summary_identity_and_counts(tmp_path: Path):
    root = tmp_path / "w0"
    for seed in (7, 17, 27):
        folder = root / f"seed{seed}"; folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(payload(1.0)))
    w003 = tmp_path / "w003.json"; w003.write_text(json.dumps(payload(3.0)))
    output = tmp_path / "out.json"
    subprocess.run([sys.executable, "scripts/summarize_response_fidelity_pareto.py",
        "--weight0-root", str(root), "--weight003-seed27", str(w003),
        "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert "not selective IPWM" in result["model_identity"]
    assert result["weight0"]["aggregate"]["response_rmse_improvement_pct"]["positive_seeds"] == 3
    assert result["weight003_seed27"]["row"]["response_rmse_improvement_pct"] < 0
