import json
from pathlib import Path

from scripts.audit_goal_completion import build


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_goal_fails_closed_without_real_evidence(tmp_path: Path) -> None:
    payload = build(tmp_path / "real", tmp_path / "score.json")
    assert payload["status"] == "INCOMPLETE"
    assert "formal original-arm Level-A Push" in payload["missing_required"]
    assert "independent ICRA/CCFA score at least 4.0/5" in payload["missing_required"]
    assert any(item["passed"] for item in payload["checks"]
               if item["name"] == "simulation contract 9/9")


def test_real_and_score_checks_require_explicit_verified_fields(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write(real / "preflight-audit.json", {
        "status": "PASS", "authorization": "LEVEL_A_TRIALS_MAY_START"})
    write(real / "schedule-completion-audit.json", {
        "status": "PASS", "changed_identity_rows": []})
    write(real / "push-summary.json", {
        "physical_feasibility_claim_level": "formal",
        "physical_feasibility_gate": {
            "counts_met": True, "raw_files_required_and_checked": True},
        "physical_feasibility_by_condition": {
            name: {"valid_trials": 10} for name in ("intact", "D2", "D3")},
    })
    score = tmp_path / "score.json"
    write(score, {"overall_score_out_of_5": 4.1,
                  "real_robot_evidence_verified": True})
    payload = build(real, score)
    names = {item["name"]: item["passed"] for item in payload["checks"]}
    assert names["real-robot preflight"]
    assert names["formal original-arm Level-A Push"]
    assert names["independent ICRA/CCFA score at least 4.0/5"]
