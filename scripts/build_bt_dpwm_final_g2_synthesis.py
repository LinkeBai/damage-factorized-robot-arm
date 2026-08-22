"""Build the single auditable G2 claim table and plotting-source CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCES = {
    "z75": "runs/g2_bt_dpwm_z75_nested_support/five_seed_development_v1/summary.json",
    "z76": "runs/g2_bt_dpwm_z76_confirmation/two_seed_confirmation_v1/summary.json",
    "z77": "runs/g2_bt_dpwm_z77_robustness/two_seed_summary_v1/summary.json",
    "z78": "runs/g2_bt_dpwm_z78_compute_failure_ledger/summary.json",
    "z79": "runs/g2_bt_dpwm_z79_uncertainty_counterfactual/calibration_v1/summary.json",
    "z80": "runs/g2_bt_dpwm_z80_context_posterior_calibration/summary.json",
    "z81": "runs/g2_bt_dpwm_z81_conformal_context_calibration/summary.json",
    "z82_z85": "runs/g2_bt_dpwm_z82_structural_ablations/two_seed_summary_v1/summary.json",
}


def load_sources(root=ROOT):
    return {key: json.loads((root / path).read_text(encoding="utf-8"))
            for key, path in SOURCES.items()}


def build(data):
    z76, z77, z79 = data["z76"], data["z77"], data["z79"]
    z81, z82 = data["z81"], data["z82_z85"]
    curves = []
    for split, payload, field in (
            ("development", data["z75"], "curves"),
            ("confirmation", z76, "curves"),
            ("robustness", z77, "aggregate_curves")):
        rows = payload[field]
        if isinstance(rows, dict):
            rows = rows.get("aggregate", rows.get("bt_dpwm", []))
        for row in rows:
            value = row.get("bt_own_gain_pct", row.get("mean"))
            if isinstance(value, dict):
                value = value["mean"]
            curves.append({"split": split, "budget": row["budget"],
                           "bt_own_gain_pct": value})
    claims = [
        {"claim": "safe nonnegative few-shot adaptation", "status": "PASS",
         "evidence": "Z76 all own gains nonnegative; max violation 0"},
        {"claim": "paired equivalence to shared at K25/K50", "status": "FAIL",
         "evidence": f"Z76 K50 lower CI {z76['gate']['paired_delta_ci_lower_bounds']['50']:.3f} pp < -1 pp"},
        {"claim": "strict robustness monotonicity", "status": "FAIL",
         "evidence": "Z77 aggregate and every-seed monotonic gates failed"},
        {"claim": "support gate rejects observed harmful proposals", "status": "PASS",
         "evidence": f"Z79 {z79['gate']['harmful_proposal_count']} / {z79['gate']['harmful_accepted_count']} harmful proposed/accepted"},
        {"claim": "raw posterior spread ranks rollout risk", "status": "FAIL",
         "evidence": f"Z79 Spearman {z79['gate']['uncertainty_risk_spearman']:.3f}"},
        {"claim": "conformal physical-context coverage", "status": "PASS",
         "evidence": f"Z81 dimensionwise MACE {z81['gate']['overall_dimensionwise_mace']:.4f}"},
        {"claim": "analytic projection enforces topology safety", "status": "PASS",
         "evidence": "Z82 no-projection violation RMSE up to 0.14454"},
        {"claim": "block-triangular object bridge is performance-dominant", "status": "NOT SUPPORTED",
         "evidence": "Z84 max paired object-RMSE effect 1.97e-5"},
    ]
    verdict = {
        "g2_artifact_delivery_complete": all(payload for payload in data.values()),
        "narrow_safe_adaptation_claim": "PASS",
        "performance_superiority_claim": "FAIL",
        "rollout_risk_calibration_claim": "FAIL",
        "physical_context_coverage_claim": "PASS",
        "advance_to_g3": "CONDITIONAL_NO_GO_PENDING_HARDWARE_READINESS_AND_CLAIM_DECISION",
    }
    return {"version": "bt_dpwm_final_g2_synthesis_v1",
            "source_artifacts": SOURCES, "claims": claims,
            "curve_rows": curves, "verdict": verdict}


def main():
    output_dir = ROOT / "runs/g2_bt_dpwm_final_synthesis_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build(load_sources())
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "claim_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("claim", "status", "evidence"))
        writer.writeheader(); writer.writerows(result["claims"])
    with (output_dir / "budget_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "budget", "bt_own_gain_pct"))
        writer.writeheader(); writer.writerows(result["curve_rows"])
    print(json.dumps(result["verdict"], indent=2))


if __name__ == "__main__":
    main()
