"""Summarize the frozen two-seed Z77 robustness matrix."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def bootstrap_ci(values, resamples, rng):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z77_robustness_matrix_v1.yaml"))
    parser.add_argument("--input-template", default=(
        "runs/g2_bt_dpwm_z77_robustness/seed{seed}_v1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z77_robustness/two_seed_summary_v1/summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = list(cfg["robustness_seeds"])
    budgets = list(cfg["transition_budgets"])
    expected = {f"{topology}__{item['name']}"
                for item in cfg["robustness_domains"]
                for topology in item["topologies"]}
    records, failures = [], []
    for seed in seeds:
        path = Path(args.input_template.format(seed=seed))
        if not path.is_file():
            failures.append({"seed": seed, "reason": "missing_summary"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        if {row["domain"] for row in rows} != expected:
            failures.append({"seed": seed, "reason": "domain_mismatch"})
            continue
        by_domain = defaultdict(list)
        for row in rows:
            by_domain[row["domain"]].append(row)
        for domain, domain_rows in by_domain.items():
            domain_rows.sort(key=lambda row: row["budget"])
            if [row["budget"] for row in domain_rows] != budgets:
                failures.append({"seed": seed, "domain": domain,
                                 "reason": "budget_mismatch"})
                continue
            bt0 = domain_rows[0]["bt_dpwm"]["overall_rmse"]
            shared0 = domain_rows[0]["shared"]["overall_rmse"]
            factor = domain.split("__robust_", 1)[1]
            for row in domain_rows:
                records.append({"seed": seed, "domain": domain, "factor": factor,
                    "budget": row["budget"],
                    "bt_own_gain_pct": 100*(bt0-row["bt_dpwm"]["overall_rmse"])/bt0,
                    "shared_own_gain_pct": 100*(shared0-row["shared"]["overall_rmse"])/shared0,
                    "bt_relative_shared_pct": row["improvement_pct"],
                    "constraint_violation_rmse": row["bt_dpwm"]["violation_rmse"]})
    complete = not failures and len(records) == len(seeds)*len(expected)*len(budgets)
    seed_curves, aggregate, factors = {}, [], []
    rng = np.random.default_rng(77077)
    if complete:
        for seed in seeds:
            seed_curves[str(seed)] = [float(np.mean([
                row["bt_own_gain_pct"] for row in records
                if row["seed"] == seed and row["budget"] == budget]))
                for budget in budgets]
        for index, budget in enumerate(budgets):
            values = [seed_curves[str(seed)][index] for seed in seeds]
            aggregate.append({"budget": budget, "bt_own_gain_pct": float(np.mean(values)),
                              "seed_values": dict(zip(map(str, seeds), values)),
                              "seed_bootstrap_95ci": bootstrap_ci(values, 10000, rng)})
        for factor in sorted({row["factor"] for row in records}):
            values = [float(np.mean([row["bt_own_gain_pct"] for row in records
                if row["seed"] == seed and row["factor"] == factor
                and row["budget"] == 50])) for seed in seeds]
            factors.append({"factor": factor, "k50_bt_own_gain_pct": float(np.mean(values)),
                            "seed_values": dict(zip(map(str, seeds), values)),
                            "seed_bootstrap_95ci": bootstrap_ci(values, 10000, rng)})
    gate = {"complete_matrix": complete, "failures": failures, "passed": False}
    if complete:
        curves = {key: np.asarray(value) for key, value in seed_curves.items()}
        aggregate_curve = np.asarray([row["bt_own_gain_pct"] for row in aggregate])
        gate.update(
            all_bt_own_gains_nonnegative=bool(min(
                row["bt_own_gain_pct"] for row in records) >= -1e-6),
            every_seed_curve_monotonic=bool(all(
                np.all(np.diff(curve) >= -1e-6) for curve in curves.values())),
            aggregate_curve_monotonic=bool(np.all(np.diff(aggregate_curve) >= -1e-6)),
            maximum_constraint_violation_rmse=float(max(
                row["constraint_violation_rmse"] for row in records)))
        gate["safety_gate_passed"] = (gate["all_bt_own_gains_nonnegative"] and
                                      gate["maximum_constraint_violation_rmse"] <= 1e-7)
        gate["passed"] = (gate["safety_gate_passed"] and
                          gate["every_seed_curve_monotonic"] and
                          gate["aggregate_curve_monotonic"])
    output = {"version": cfg["version"], "statistical_unit": "seed",
              "seeds": seeds, "domains": sorted(expected), "budgets": budgets,
              "aggregate_curves": aggregate, "factor_k50": factors,
              "records": records, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
