"""Frozen three-seed fair Z70 comparison against shared h136/240."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    seeds, budgets = (7, 17, 27), None
    bt_curves, shared_curves, relative_curves = [], [], []
    seed_rows, violations, object_changes = [], [], []
    for seed in seeds:
        path = Path(f"runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_gate_v2/summary.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for row in payload["rows"]:
            grouped[row["domain"]].append(row)
        domains = {}
        for domain, rows in grouped.items():
            rows.sort(key=lambda x: x["budget"]); budgets = [x["budget"] for x in rows]
            bt0, shared0 = rows[0]["bt_dpwm"], rows[0]["shared"]
            bt = [100*(bt0["overall_rmse"]-x["bt_dpwm"]["overall_rmse"])
                  / bt0["overall_rmse"] for x in rows]
            shared = [100*(shared0["overall_rmse"]-x["shared"]["overall_rmse"])
                      / shared0["overall_rmse"] for x in rows]
            relative = [x["improvement_pct"] for x in rows]
            obj = [100*(bt0["object_rmse"]-x["bt_dpwm"]["object_rmse"])
                   / max(bt0["object_rmse"], 1e-12) for x in rows]
            violation = [x["bt_dpwm"]["violation_rmse"] for x in rows]
            bt_curves.append(bt); shared_curves.append(shared); relative_curves.append(relative)
            violations.extend(violation); object_changes.extend(obj)
            domains[domain] = {"bt_own_gain_pct": bt,
                "shared_own_gain_pct": shared, "bt_relative_shared_pct": relative,
                "object_change_pct": obj, "constraint_violation_rmse": violation}
        seed_rows.append({"seed": seed, "domains": domains})
    bt_mean = np.asarray(bt_curves).mean(0)
    shared_mean = np.asarray(shared_curves).mean(0)
    relative_mean = np.asarray(relative_curves).mean(0)
    gate = {
        "all_bt_own_gains_nonnegative": bool(np.min(bt_curves) >= -1e-6),
        "aggregate_bt_curve_monotonic": bool(np.all(np.diff(bt_mean) >= -1e-6)),
        "bt_k10_k25_sample_efficiency_exceeds_shared": bool(
            bt_mean[2] >= shared_mean[2] and bt_mean[3] >= shared_mean[3]),
        "final_mean_relative_shared_pct": float(relative_mean[-1]),
        "maximum_constraint_violation_rmse": max(violations),
        "maximum_absolute_object_change_pct": max(abs(x) for x in object_changes),
    }
    gate["passed"] = (gate["all_bt_own_gains_nonnegative"] and
        gate["aggregate_bt_curve_monotonic"] and
        gate["bt_k10_k25_sample_efficiency_exceeds_shared"] and
        gate["final_mean_relative_shared_pct"] >= -1.0 and
        gate["maximum_constraint_violation_rmse"] <= 1e-7)
    result = {"version": "g2_bt_dpwm_z70_fair_three_seed_gate_v1",
        "budgets": budgets, "bt_mean_own_gain_pct": bt_mean.tolist(),
        "shared_mean_own_gain_pct": shared_mean.tolist(),
        "bt_mean_relative_shared_pct": relative_mean.tolist(),
        "seeds": seed_rows, "gate": gate}
    output = Path("runs/g2_bt_dpwm_z69_adapter_z70/three_seed_fair_gate_v1/summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**gate, "bt_mean": bt_mean.tolist(),
                      "shared_mean": shared_mean.tolist()}, indent=2))


if __name__ == "__main__":
    main()
