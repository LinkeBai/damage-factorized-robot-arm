"""Fail-closed completion audit for the frozen ICRA paper objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def build(real_root: Path, score_path: Path) -> dict:
    simulation = load(ROOT / "results/final/primary-evidence-contract-audit.json")
    provenance = load(ROOT / "results/final/primary-result-provenance-ledger.json")
    advantages = load(ROOT / "results/final/large-advantage-metric-audit.json")
    pdf = load(ROOT / "results/final/icra-pdf-anonymity-audit.json")
    preflight = load(real_root / "preflight-audit.json")
    schedule = load(real_root / "schedule-completion-audit.json")
    push = load(real_root / "push-summary.json")
    grasp = load(real_root / "grasp-feasibility-summary.json")
    score = load(score_path)

    checks = []
    def check(name: str, passed: bool, evidence: str, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed),
                       "required": required, "evidence": evidence})

    check("simulation contract 9/9", bool(simulation)
          and simulation.get("implemented_cells") == simulation.get("required_cells") == 9
          and simulation.get("same_protocol_result_cells") == 9,
          "results/final/primary-evidence-contract-audit.json")
    check("model identity/provenance ledger", bool(provenance),
          "results/final/primary-result-provenance-ledger.json")
    check("large advantages have explicit attribution boundary", bool(advantages)
          and "global residual" in advantages.get("headline_pair", {}).get("boundary", "")
          and advantages.get("selection_policy", {}).get("all_seeds_reported") is True,
          "results/final/large-advantage-metric-audit.json")
    check("paper PDF anonymity/layout audit", bool(pdf) and pdf.get("status") == "PASS"
          and pdf.get("pages", 99) <= pdf.get("max_pages", 0),
          "results/final/icra-pdf-anonymity-audit.json")

    check("real-robot preflight", bool(preflight) and preflight.get("status") == "PASS"
          and preflight.get("authorization") in {
              "LEVEL_A_TRIALS_MAY_START", "LEVEL_B_METHOD_TRIALS_MAY_START"},
          display_path(real_root / "preflight-audit.json"))
    check("frozen/completed schedule identity", bool(schedule)
          and schedule.get("status") == "PASS"
          and not schedule.get("changed_identity_rows"),
          display_path(real_root / "schedule-completion-audit.json"))
    feasibility = (push or {}).get("physical_feasibility_gate", {})
    check("formal original-arm Level-A Push", bool(push)
          and push.get("physical_feasibility_claim_level") == "formal"
          and feasibility.get("counts_met") is True
          and feasibility.get("raw_files_required_and_checked") is True,
          display_path(real_root / "push-summary.json"))
    check("real Push includes intact/D2/D3", bool(push)
          and all((push.get("physical_feasibility_by_condition", {}).get(name, {})
                   .get("valid_trials", 0) >= 10) for name in ("intact", "D2", "D3")),
          display_path(real_root / "push-summary.json"))
    check("fixed-pregrasp grasp feasibility", bool(grasp),
          display_path(real_root / "grasp-feasibility-summary.json"), required=False)
    check("independent ICRA/CCFA score at least 4.0/5", bool(score)
          and float(score.get("overall_score_out_of_5", 0)) >= 4.0
          and score.get("real_robot_evidence_verified") is True,
          display_path(score_path))

    required = [item for item in checks if item["required"]]
    passed = all(item["passed"] for item in required)
    missing = [item["name"] for item in required if not item["passed"]]
    return {
        "objective": "frozen ICRA fault-adaptation and control-diagnosis paper",
        "status": "COMPLETE" if passed else "INCOMPLETE",
        "required_checks_passed": sum(item["passed"] for item in required),
        "required_checks_total": len(required),
        "checks": checks,
        "missing_required": missing,
        "completion_rule": (
            "COMPLETE requires every required check; optional grasp cannot compensate "
            "for missing Push, provenance, paper, or score evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path,
                        default=ROOT / "results/real_robot")
    parser.add_argument("--score", type=Path,
                        default=ROOT / "results/final/independent-final-score.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/final/goal-completion-audit.json")
    args = parser.parse_args()
    payload = build(args.real_root, args.score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
