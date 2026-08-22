"""Create the preregistered five-seed Z71 evidence table and seed bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def bootstrap_ci(values, resamples, confidence, rng):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1)
    alpha = (1-confidence)/2
    return [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1-alpha))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_five_seed_completion_z71_v1.yaml"))
    parser.add_argument("--input-template", default=(
        "runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_gate_v2/summary.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z71_five_seed/five_seed_gate_v1/summary.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds, expected_domains = config["seeds"], set(config["domains"])
    budgets = list(config["transition_budgets"])
    records, failures = [], []
    for seed in seeds:
        path = Path(args.input_template.format(seed=seed))
        if not path.is_file():
            failures.append({"seed": seed, "reason": "missing_summary", "path": str(path)})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["rows"]
            actual_domains = {row["domain"] for row in rows}
            if actual_domains != expected_domains:
                raise ValueError(f"domain mismatch: {sorted(actual_domains)}")
            for domain in sorted(expected_domains):
                domain_rows = sorted((row for row in rows if row["domain"] == domain),
                                     key=lambda row: row["budget"])
                if [row["budget"] for row in domain_rows] != budgets:
                    raise ValueError(f"budget mismatch for {domain}")
                bt0, shared0 = domain_rows[0]["bt_dpwm"], domain_rows[0]["shared"]
                for row in domain_rows:
                    bt, shared = row["bt_dpwm"], row["shared"]
                    records.append({"seed": seed, "domain": domain, "budget": row["budget"],
                        "bt_own_gain_pct": 100*(bt0["overall_rmse"]-bt["overall_rmse"])/bt0["overall_rmse"],
                        "shared_own_gain_pct": 100*(shared0["overall_rmse"]-shared["overall_rmse"])/shared0["overall_rmse"],
                        "bt_relative_shared_pct": float(row["improvement_pct"]),
                        "bt_free_rmse": bt["free_rmse"], "bt_object_rmse": bt["object_rmse"],
                        "bt_overall_rmse": bt["overall_rmse"],
                        "shared_free_rmse": shared["free_rmse"],
                        "shared_object_rmse": shared["object_rmse"],
                        "shared_overall_rmse": shared["overall_rmse"],
                        "constraint_violation_rmse": bt["violation_rmse"]})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"seed": seed, "reason": str(exc), "path": str(path)})
    complete = not failures and len({row["seed"] for row in records}) == len(seeds)
    grouped, seed_budget = defaultdict(list), defaultdict(list)
    for row in records:
        for metric in ("bt_own_gain_pct", "shared_own_gain_pct", "bt_relative_shared_pct"):
            grouped[(row["budget"], metric)].append(row[metric])
        seed_budget[(row["seed"], row["budget"])].append(row)
    rng = np.random.default_rng(71071)
    curves = []
    if complete:
        for budget in budgets:
            entry = {"budget": budget}
            for metric in ("bt_own_gain_pct", "shared_own_gain_pct", "bt_relative_shared_pct"):
                seed_values = [float(np.mean([row[metric] for row in seed_budget[(seed, budget)]]))
                               for seed in seeds]
                entry[metric] = {"mean": float(np.mean(seed_values)),
                    "seed_values": dict(zip(map(str, seeds), seed_values)),
                    "seed_bootstrap_95ci": bootstrap_ci(seed_values,
                        int(config["bootstrap_resamples"]), float(config["confidence_level"]), rng)}
            efficiency_delta = [
                entry["bt_own_gain_pct"]["seed_values"][str(seed)]-
                entry["shared_own_gain_pct"]["seed_values"][str(seed)]
                for seed in seeds]
            entry["bt_minus_shared_own_gain_pct_points"] = {
                "mean": float(np.mean(efficiency_delta)),
                "seed_values": dict(zip(map(str, seeds), efficiency_delta)),
                "seed_bootstrap_95ci": bootstrap_ci(efficiency_delta,
                    int(config["bootstrap_resamples"]),
                    float(config["confidence_level"]), rng)}
            curves.append(entry)
    gate = {"complete_five_seed_matrix": complete,
            "failures_recorded": failures,
            "passed": False}
    if complete:
        bt_curve = np.asarray([x["bt_own_gain_pct"]["mean"] for x in curves])
        shared_curve = np.asarray([x["shared_own_gain_pct"]["mean"] for x in curves])
        relative = np.asarray([x["bt_relative_shared_pct"]["mean"] for x in curves])
        gate.update(all_bt_own_gains_nonnegative=bool(min(
            row["bt_own_gain_pct"] for row in records) >= -1e-6),
            aggregate_bt_curve_monotonic=bool(np.all(np.diff(bt_curve) >= -1e-6)),
            k10_k25_sample_efficiency_exceeds_shared=bool(
                bt_curve[2] >= shared_curve[2] and bt_curve[3] >= shared_curve[3]),
            final_mean_relative_shared_pct=float(relative[-1]),
            maximum_constraint_violation_rmse=float(max(
                row["constraint_violation_rmse"] for row in records)))
        criteria = config["gate"]
        gate["passed"] = (gate["all_bt_own_gains_nonnegative"] and
            gate["aggregate_bt_curve_monotonic"] and
            gate["k10_k25_sample_efficiency_exceeds_shared"] and
            gate["final_mean_relative_shared_pct"] >=
                float(criteria["minimum_k50_relative_shared_pct"]) and
            gate["maximum_constraint_violation_rmse"] <=
                float(criteria["maximum_constraint_violation_rmse"]))
    output = {"version": config["version"], "statistical_unit": "seed",
              "bootstrap_resamples": config["bootstrap_resamples"],
              "seeds": seeds, "domains": sorted(expected_domains), "budgets": budgets,
              "curves": curves, "records": records, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
