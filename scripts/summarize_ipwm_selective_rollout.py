"""Summarize the frozen selective-IPWM audit across random seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTIVE_DOMAINS = {
    "D3__high_damping",
    "D3__mixed_composition",
    "D3__mixed_unseen",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    cells = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyed = {
            (row["domain"], int(row["horizon"]), row["method"]): row
            for row in payload["rows"]
        }
        for domain in ACTIVE_DOMAINS:
            for horizon in (10, 25, 50):
                carrier = keyed[(domain, horizon, "carrier_no_intervention")]
                method = keyed[(domain, horizon, "selective_ipwm")]
                cells.append({
                    "seed": int(payload["seed"]),
                    "domain": domain,
                    "horizon": horizon,
                    "object_improvement_pct": 100.0 * (
                        carrier["object_rmse"] - method["object_rmse"]
                    ) / carrier["object_rmse"],
                    "free_change_pct": 100.0 * (
                        method["free_rmse"] - carrier["free_rmse"]
                    ) / carrier["free_rmse"],
                    "pusher_change_pct": 100.0 * (
                        method["pusher_xy_rmse"] - carrier["pusher_xy_rmse"]
                    ) / carrier["pusher_xy_rmse"],
                    "violation_rmse": method["violation_rmse"],
                })

    seeds = sorted({row["seed"] for row in cells})
    seed_means = np.asarray([
        np.mean([row["object_improvement_pct"] for row in cells if row["seed"] == seed])
        for seed in seeds
    ])
    rng = np.random.default_rng(args.seed)
    bootstrap = seed_means[
        rng.integers(0, len(seed_means), size=(args.bootstrap_samples, len(seed_means)))
    ].mean(axis=1)
    summary = {
        "version": "g2_ipwm_selective_rollout_summary_v1",
        "comparison": "selective_ipwm_vs_mechanism_matched_carrier",
        "seeds": seeds,
        "active_domains": sorted(ACTIVE_DOMAINS),
        "horizons": [10, 25, 50],
        "cell_count": len(cells),
        "object_improvement_pct": {
            "cell_min": float(min(row["object_improvement_pct"] for row in cells)),
            "cell_max": float(max(row["object_improvement_pct"] for row in cells)),
            "seed_means": dict(zip(map(str, seeds), map(float, seed_means))),
            "seed_mean": float(seed_means.mean()),
            "seed_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "positive_cells": sum(row["object_improvement_pct"] > 0 for row in cells),
        },
        "safety": {
            "max_abs_free_change_pct": float(max(abs(row["free_change_pct"]) for row in cells)),
            "max_abs_pusher_change_pct": float(max(abs(row["pusher_change_pct"]) for row in cells)),
            "max_violation_rmse": float(max(row["violation_rmse"] for row in cells)),
        },
        "cells": cells,
        "limitations": [
            "The seed-level bootstrap contains only three seeds.",
            "The physical-support threshold was frozen after seed 27, but the state-isolation wrapper was designed after retrospective inspection of seeds 37 and 47.",
            "These are open-loop prediction results, not closed-loop task results.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
