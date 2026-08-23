"""Build the machine-readable frozen G2-R0 physical-context gate ledger."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("runs/g2_r0_physical_context_residual")
RUNS = {
    7: ROOT / "seed7_centered_regression_v1",
    17: ROOT / "seed17_centered_v3",
    27: ROOT / "seed27_confirmation_v1",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def improvement(rows, domain, horizon, metric="object_rmse"):
    by_method = {row["method"]: row for row in rows
                 if row["domain"] == domain and row["horizon"] == horizon}
    base, model = by_method["shared_projected"], by_method["strict_bt"]
    return 100.0 * (base[metric] - model[metric]) / base[metric]


def main():
    domains = ("D3__mixed_composition", "D3__mixed_unseen")
    horizons = (10, 25, 50)
    seeds = {}
    all_object = []
    for seed, run in RUNS.items():
        k25_name = ("k25_frozen_d3_metrics.json" if seed == 27 else
                    "k25_scale_1.38_delayed_start8_d3_metrics.json")
        k25 = load(run / k25_name)
        k0 = load(run / ("k0_frozen_d3_metrics.json" if seed != 7 else
                         "k0_d3_metrics.json"))
        cells = []
        for domain in domains:
            for horizon in horizons:
                row = {"domain": domain, "horizon": horizon}
                for metric in ("object_rmse", "free_rmse", "overall_rmse",
                               "pusher_xy_rmse"):
                    row[metric.replace("_rmse", "_improvement_pct")] = improvement(
                        k25["rows"], domain, horizon, metric)
                row["constraint_violation_rmse"] = max(
                    item["violation_rmse"] for item in k25["rows"]
                    if item["domain"] == domain and item["horizon"] == horizon
                    and item["method"] == "strict_bt")
                row["k0_object_improvement_pct"] = improvement(
                    k0["rows"], domain, horizon)
                cells.append(row); all_object.append(row["object_improvement_pct"])
        seeds[str(seed)] = {"cells": cells,
            "all_object_positive": all(x > 0.0 for x in
                                       [r["object_improvement_pct"] for r in cells]),
            "all_object_above_2pct": all(x >= 2.0 for x in
                                        [r["object_improvement_pct"] for r in cells])}
    controls = {}
    for seed, run in RUNS.items():
        name = ("k25_frozen_controls_metrics.json" if seed == 27 else
                "k25_frozen_iid_routing_fixed_metrics.json")
        data = load(run / name)
        maximum = 0.0
        for row in data["rows"]:
            if row["method"] != "strict_bt": continue
            maximum = max(maximum, abs(improvement(
                data["rows"], row["domain"], row["horizon"])))
        controls[str(seed)] = {"maximum_abs_object_difference_pct": maximum}
    result = {
        "version": "g2_r0_physical_context_frozen_gate_v1",
        "frozen_rule": {"posterior": "Z65", "budget": 25,
                        "posterior_scale": 1.38, "grace_steps": 8,
                        "depth_ramp": 0.06, "zero_context_exact": True},
        "seeds": seeds, "controls": controls,
        "development_seeds_all_above_2pct": all(
            seeds[str(s)]["all_object_above_2pct"] for s in (7, 17)),
        "confirmation_all_positive": seeds["27"]["all_object_positive"],
        "confirmation_all_above_2pct": seeds["27"]["all_object_above_2pct"],
        "minimum_confirmation_object_improvement_pct": min(
            row["object_improvement_pct"] for row in seeds["27"]["cells"]),
        "zero_constraint_violations": all(
            row["constraint_violation_rmse"] == 0.0
            for seed in seeds.values() for row in seed["cells"]),
    }
    output = ROOT / "frozen_gate_summary_v1.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
