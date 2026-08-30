"""Summarize the frozen observable physical-support router audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE = "shared_matched_adapter"
FULL = "bt_matched_adapter"
FALLBACK = "bt_no_intervention_matched_adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="27,37,47")
    parser.add_argument("--development-seed", type=int, default=27)
    parser.add_argument("--threshold", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    rows = []
    for seed in seeds:
        payload = json.loads(
            (args.root / f"seed{seed}" / "metrics.json").read_text(encoding="utf-8")
        )
        lookup = {
            (row["domain"], int(row["horizon"]), row["method"]): row
            for row in payload["rows"]
        }
        norms = {
            item["domain"]: float(np.linalg.norm(item["context"]))
            for item in payload["context_diagnostics"]
        }
        for domain, norm in sorted(norms.items()):
            route = FULL if norm >= args.threshold else FALLBACK
            for horizon in (10, 25, 50):
                base = lookup[(domain, horizon, BASE)]
                chosen = lookup[(domain, horizon, route)]
                rows.append({
                    "seed": seed,
                    "role": "development" if seed == args.development_seed else "confirmation",
                    "domain": domain,
                    "horizon": horizon,
                    "context_norm": norm,
                    "route": "full" if route == FULL else "fallback",
                    "object_improvement_pct": 100.0 * (
                        base["object_rmse"] - chosen["object_rmse"]
                    ) / base["object_rmse"],
                    "free_change_pct": 100.0 * (
                        chosen["free_rmse"] - base["free_rmse"]
                    ) / base["free_rmse"],
                    "pusher_change_pct": 100.0 * (
                        chosen["pusher_xy_rmse"] - base["pusher_xy_rmse"]
                    ) / base["pusher_xy_rmse"],
                    "violation_rmse": chosen["violation_rmse"],
                })

    confirmation = [row for row in rows if row["role"] == "confirmation"]
    summary = {
        "version": "g2_ipwm_physical_support_gate_summary_v1",
        "threshold": args.threshold,
        "development_seed": args.development_seed,
        "confirmation_seeds": [s for s in seeds if s != args.development_seed],
        "confirmation_cells": len(confirmation),
        "confirmation_nonnegative_object_cells": sum(
            row["object_improvement_pct"] >= 0.0 for row in confirmation
        ),
        "confirmation_within_2pct_object_nonregression": sum(
            row["object_improvement_pct"] >= -2.0 for row in confirmation
        ),
        "confirmation_min_object_improvement_pct": min(
            row["object_improvement_pct"] for row in confirmation
        ),
        "confirmation_max_free_regression_pct": max(
            row["free_change_pct"] for row in confirmation
        ),
        "all_locked_violations_zero": all(
            row["violation_rmse"] == 0.0 for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
