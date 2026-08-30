"""Analyze counterfactual rollout risk versus BT context posterior spread."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


def correlation(rows):
    if len(rows) < 3:
        return {"n": len(rows), "spearman": None, "pvalue": None}
    result = spearmanr([row["context_mean_std"] for row in rows],
                       [row["candidate_harm_pct"] for row in rows])
    return {"n": len(rows), "spearman": float(result.statistic),
            "pvalue": float(result.pvalue)}


def coverage_curve(rows, levels):
    ordered = sorted(rows, key=lambda row: row["context_mean_std"])
    curve = []
    for coverage in levels:
        count = max(1, int(np.ceil(len(ordered)*float(coverage))))
        retained = ordered[:count]
        harms = np.asarray([row["candidate_harm_pct"] for row in retained])
        curve.append({"coverage": float(coverage), "retained": count,
                      "mean_candidate_harm_pct": float(harms.mean()),
                      "harmful_fraction": float(np.mean(harms > 1e-9)),
                      "worst_candidate_harm_pct": float(harms.max())})
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z79_uncertainty_counterfactual_v1.yaml"))
    parser.add_argument("--input-template", default=(
        "runs/g2_bt_dpwm_z79_uncertainty_counterfactual/seed{seed}_v1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z79_uncertainty_counterfactual/calibration_v1/summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records, failures = [], []
    for seed in cfg["calibration_seeds"]:
        path = Path(args.input_template.format(seed=seed))
        if not path.is_file():
            failures.append({"seed": seed, "reason": "missing_summary"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for row in payload["rows"]:
            grouped[row["domain"]].append(row)
        for domain, rows in grouped.items():
            rows.sort(key=lambda row: row["budget"])
            previous_rmse = rows[0]["bt_dpwm"]["overall_rmse"]
            for row in rows[1:]:
                diag = row["bt_adaptation"]
                candidate = row.get("bt_candidate_counterfactual")
                if (candidate is not None and
                        diag.get("context_log_variance") is not None and
                        diag.get("raw_z_norm", 0.0) > 0.0):
                    harm = 100*(candidate["overall_rmse"]-previous_rmse)/previous_rmse
                    records.append({"seed": seed, "domain": domain,
                        "topology": domain.split("__", 1)[0],
                        "budget": row["budget"],
                        "context_mean_std": diag["context_mean_std"],
                        "support_validation_improvement": diag["validation_improvement"],
                        "candidate_harm_pct": harm,
                        "candidate_harmful": bool(harm > 1e-9),
                        "candidate_accepted": bool(not diag["rolled_back"])})
                previous_rmse = row["bt_dpwm"]["overall_rmse"]
    complete = not failures and len(records) > 0
    strata = {"overall": correlation(records)}
    for budget in sorted({row["budget"] for row in records}):
        strata[f"budget_{budget}"] = correlation(
            [row for row in records if row["budget"] == budget])
    for topology in sorted({row["topology"] for row in records}):
        strata[f"topology_{topology}"] = correlation(
            [row for row in records if row["topology"] == topology])
    curve = coverage_curve(records, cfg["coverage_levels"]) if complete else []
    harmful = [row for row in records if row["candidate_harmful"]]
    accepted = [row for row in records if row["candidate_accepted"]]
    gate = {"complete": complete, "failures": failures, "passed": False}
    if complete:
        overall = strata["overall"]
        gate.update(
            proposal_count=len(records), harmful_proposal_count=len(harmful),
            accepted_proposal_count=len(accepted),
            harmful_accepted_count=sum(row["candidate_harmful"] for row in accepted),
            support_gate_rejected_all_harmful=bool(all(
                not row["candidate_accepted"] for row in harmful)),
            uncertainty_risk_spearman=overall["spearman"],
            uncertainty_risk_pvalue=overall["pvalue"],
            uncertainty_risk_ranking_passed=bool(
                overall["spearman"] > 0 and overall["pvalue"] < 0.05))
        gate["passed"] = (gate["support_gate_rejected_all_harmful"] and
                          gate["uncertainty_risk_ranking_passed"])
    output = {"version": cfg["version"], "statistical_unit": "proposal",
              "seeds": cfg["calibration_seeds"], "records": records,
              "stratified_calibration": strata, "coverage_risk": curve,
              "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
