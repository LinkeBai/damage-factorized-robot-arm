"""Requirement-by-requirement audit of the frozen BT-DPWM G2 evidence package."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
FILES = {
    "development": "runs/g2_bt_dpwm_z75_nested_support/five_seed_development_v1/summary.json",
    "confirmation": "runs/g2_bt_dpwm_z76_confirmation/two_seed_confirmation_v1/summary.json",
    "robustness": "runs/g2_bt_dpwm_z77_robustness/two_seed_summary_v1/summary.json",
    "compute_failure": "runs/g2_bt_dpwm_z78_compute_failure_ledger/summary.json",
    "rollout_risk": "runs/g2_bt_dpwm_z79_uncertainty_counterfactual/calibration_v1/summary.json",
    "gaussian_coverage": "runs/g2_bt_dpwm_z80_context_posterior_calibration/summary.json",
    "conformal_coverage": "runs/g2_bt_dpwm_z81_conformal_context_calibration/summary.json",
    "structural_ablations": "runs/g2_bt_dpwm_z82_structural_ablations/two_seed_summary_v1/summary.json",
    "synthesis": "runs/g2_bt_dpwm_final_synthesis_v1/summary.json",
    "claim_table": "runs/g2_bt_dpwm_final_synthesis_v1/claim_table.csv",
    "curve_source": "runs/g2_bt_dpwm_final_synthesis_v1/budget_curves.csv",
}


def _tracked(relative):
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def audit():
    payload = {}
    checks = []
    for name, relative in FILES.items():
        path = ROOT / relative
        checks.append({"requirement": f"artifact:{name}", "passed": path.is_file(),
                       "evidence": relative})
        checks.append({"requirement": f"git_tracked:{name}", "passed": _tracked(relative),
                       "evidence": relative})
        if path.suffix == ".json" and path.is_file():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))

    z75, z76 = payload["development"], payload["confirmation"]
    z77, z78 = payload["robustness"], payload["compute_failure"]
    z79, z80 = payload["rollout_risk"], payload["gaussian_coverage"]
    z81, z82 = payload["conformal_coverage"], payload["structural_ablations"]
    requirements = [
        ("Z71 failures and Z72-Z75 trajectory retained", z78["all_failures_retained"] and
         len(z78["failure_runs"]) >= 10, "Z78 failure ledger"),
        ("five-seed development matrix complete", z75["gate"]["complete_five_seed_matrix"], "Z75"),
        ("independent confirmation matrix complete", z76["gate"]["complete_five_seed_matrix"], "Z76"),
        ("confirmation has no negative own gain", z76["gate"]["all_bt_own_gains_nonnegative"], "Z76"),
        ("confirmation aggregate budget curve monotonic", z76["gate"]["aggregate_bt_curve_monotonic"], "Z76"),
        ("confirmation constraint violation is zero", z76["gate"]["maximum_constraint_violation_rmse"] <= 1e-7, "Z76"),
        ("K>0 has independent positive contribution", any(
            row["budget"] > 0 and row["bt_own_gain_pct"]["mean"] > 0 for row in z76["curves"]), "Z76"),
        ("paired seed CI and equivalence decision recorded", "paired_delta_ci_lower_bounds" in z76["gate"], "Z76"),
        ("six-factor robustness complete", len(z77["factor_k50"]) == 6 and z77["gate"]["complete_matrix"], "Z77"),
        ("robustness safety gate passes", z77["gate"]["safety_gate_passed"], "Z77"),
        ("stratified coverage-risk audit complete", bool(z79["stratified_calibration"]) and bool(z79["coverage_risk"]), "Z79"),
        ("rollout-risk failure disclosed", not z79["gate"]["uncertainty_risk_ranking_passed"], "Z79"),
        ("Gaussian context coverage recorded", z80["gate"]["passed"], "Z80"),
        ("conformal context coverage passes", z81["gate"]["passed"], "Z81"),
        ("all four BT structural conclusions recorded", all(z82["conclusions"].values()), "Z82-Z85"),
        ("parameter comparison recorded", bool(z78["parameters"]), "Z78"),
        ("training/adaptation wall-clock recorded", bool(z78["wall_clock_stages"]) and bool(z78["wall_clock_total_s"]), "Z78"),
        ("unified synthesis explicitly discloses failed strong claims",
         payload["synthesis"]["verdict"]["performance_superiority_claim"] == "FAIL" and
         payload["synthesis"]["verdict"]["rollout_risk_calibration_claim"] == "FAIL", "final synthesis"),
    ]
    checks.extend({"requirement": name, "passed": bool(passed), "evidence": evidence}
                  for name, passed, evidence in requirements)
    return {"version": "bt_dpwm_g2_completion_audit_v1", "checks": checks,
            "passed_count": sum(row["passed"] for row in checks),
            "check_count": len(checks), "passed": all(row["passed"] for row in checks)}


def main():
    result = audit()
    output = ROOT / "runs/g2_bt_dpwm_final_synthesis_v1/completion_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed_count", "check_count", "passed")}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
