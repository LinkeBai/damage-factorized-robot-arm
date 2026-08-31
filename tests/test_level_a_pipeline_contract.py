from pathlib import Path


def test_level_a_pipeline_contains_every_fail_closed_stage() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_real_robot_level_a_pipeline.ps1").read_text(
        encoding="utf-8")
    required = [
        "audit_level_a_trajectory_library.py",
        "audit_real_robot_preflight.py",
        "--mode level_a",
        "audit_real_robot_schedule_completion.py",
        "analyze_real_robot_push.py",
        "--require-files",
        "build_real_robot_feasibility_assets.py",
        "real-robot-feasibility.pdf",
        "real-robot-feasibility-table.tex",
    ]
    assert all(item in text for item in required)
    assert text.count("$LASTEXITCODE -ne 0") == 5
