"""Aggregate the frozen three-seed Z65 deployment gate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    ap.add_argument("--run-template", default=(
        "runs/g2_bt_dpwm_context_encoder_z65/seed{seed}_gate_v1/summary.json"))
    ap.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_context_encoder_z65/three_seed_gate_v1/summary.json"))
    args = ap.parse_args()
    seed_summaries, all_curves, object_changes, violations = [], [], [], []
    for seed in args.seeds:
        source = Path(args.run_template.format(seed=seed))
        payload = json.loads(source.read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for row in payload["rows"]:
            grouped[row["domain"]].append(row)
        domains, curves = {}, []
        for domain, rows in grouped.items():
            rows.sort(key=lambda x: x["budget"])
            base = rows[0]["bt_dpwm"]
            curve = [100.0*(base["overall_rmse"]-x["bt_dpwm"]["overall_rmse"])
                     / base["overall_rmse"] for x in rows]
            object_curve = [100.0*(base["object_rmse"]-x["bt_dpwm"]["object_rmse"])
                            / max(base["object_rmse"], 1e-12) for x in rows]
            constraint = [x["bt_dpwm"]["violation_rmse"] for x in rows]
            domains[domain] = {"budgets": [x["budget"] for x in rows],
                "bt_own_gain_pct": curve, "object_change_pct": object_curve,
                "constraint_violation_rmse": constraint}
            curves.append(curve); object_changes.extend(object_curve)
            violations.extend(constraint)
        mean_curve = np.asarray(curves).mean(0).tolist()
        all_curves.extend(curves)
        seed_summaries.append({"seed": seed, "domains": domains,
            "mean_bt_own_gain_pct": mean_curve,
            "nonnegative": min(map(min, curves)) >= -1e-6,
            "mean_monotonic": bool(np.all(np.diff(mean_curve) >= -1e-6)),
            "positive_domains_at_final_budget": sum(x[-1] > 1e-6 for x in curves)})
    aggregate = np.asarray(all_curves)
    gate = {
        "all_seed_domain_budgets_nonnegative": bool(aggregate.min() >= -1e-6),
        "all_seed_mean_curves_monotonic": all(x["mean_monotonic"] for x in seed_summaries),
        "all_seeds_at_least_three_positive_domains": all(
            x["positive_domains_at_final_budget"] >= 3 for x in seed_summaries),
        "maximum_constraint_violation_rmse": max(violations),
        "maximum_absolute_object_change_pct": max(abs(x) for x in object_changes),
    }
    gate["passed"] = (gate["all_seed_domain_budgets_nonnegative"] and
        gate["all_seed_mean_curves_monotonic"] and
        gate["all_seeds_at_least_three_positive_domains"] and
        gate["maximum_constraint_violation_rmse"] <= 1e-7)
    output = {"version": "g2_bt_dpwm_context_encoder_z65_three_seed_gate_v1",
              "seeds": seed_summaries, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
