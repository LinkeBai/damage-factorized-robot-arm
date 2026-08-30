"""Summarize frozen IPWM zero-shot transfer to calibrated GenkiArm Push."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_interval(values: np.ndarray, *, seed: int = 20260828) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(paths: list[Path]) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    routing: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        routing.extend({"seed": seed, **row} for row in payload["routing"])
        indexed = {(row["method"], int(row["horizon"])): row for row in payload["rows"]}
        for horizon in (10, 25, 50):
            carrier = indexed[("carrier_no_intervention", horizon)]
            selective = indexed[("selective_ipwm", horizon)]
            routed = indexed[("routed_selective_ipwm", horizon)]
            baseline = float(carrier["object_rmse"])
            cells.append({
                "seed": seed,
                "horizon": horizon,
                "carrier_object_rmse": baseline,
                "selective_object_rmse": float(selective["object_rmse"]),
                "routed_object_rmse": float(routed["object_rmse"]),
                "selective_improvement_pct": 100.0 * (
                    baseline - float(selective["object_rmse"])
                ) / baseline,
                "routed_improvement_pct": 100.0 * (
                    baseline - float(routed["object_rmse"])
                ) / baseline,
                "selective_free_change": float(selective["free_rmse"])
                - float(carrier["free_rmse"]),
                "routed_free_change": float(routed["free_rmse"])
                - float(carrier["free_rmse"]),
                "selective_violation_rmse": float(selective["violation_rmse"]),
                "routed_violation_rmse": float(routed["violation_rmse"]),
            })
    seeds = sorted({int(cell["seed"]) for cell in cells})
    selective_means = np.array([
        np.mean([float(c["selective_improvement_pct"]) for c in cells if c["seed"] == seed])
        for seed in seeds
    ])
    routed_means = np.array([
        np.mean([float(c["routed_improvement_pct"]) for c in cells if c["seed"] == seed])
        for seed in seeds
    ])
    return {
        "version": "g2_ipwm_genkiarm_zero_shot_summary_v1",
        "status": "exploratory_actual_model_transfer_audit",
        "seeds": seeds,
        "routing": routing,
        "cells": cells,
        "selective": {
            "positive_cells": sum(float(c["selective_improvement_pct"]) > 0 for c in cells),
            "negative_cells": sum(float(c["selective_improvement_pct"]) < 0 for c in cells),
            "seed_mean_improvement_pct": selective_means.tolist(),
            "mean_improvement_pct": float(selective_means.mean()),
            "seed_bootstrap_95_ci": bootstrap_interval(selective_means),
        },
        "routed": {
            "positive_cells": sum(float(c["routed_improvement_pct"]) > 1e-10 for c in cells),
            "tie_cells": sum(abs(float(c["routed_improvement_pct"])) <= 1e-10 for c in cells),
            "negative_cells": sum(float(c["routed_improvement_pct"]) < -1e-10 for c in cells),
            "seed_mean_improvement_pct": routed_means.tolist(),
            "mean_improvement_pct": float(routed_means.mean()),
            "seed_bootstrap_95_ci": bootstrap_interval(routed_means),
        },
        "safety": {
            "max_abs_routed_free_rmse_change": max(abs(float(c["routed_free_change"])) for c in cells),
            "max_routed_violation_rmse": max(float(c["routed_violation_rmse"]) for c in cells),
        },
        "decision": "partial_prediction_transfer_not_deployment_go",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
