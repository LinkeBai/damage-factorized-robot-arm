"""Summarize paired weight-10 versus weight-0 decision-loss ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reduction(base: float, candidate: float) -> float:
    return 100.0 * (base - candidate) / base


def metrics(summary: dict, method: str) -> dict:
    value = summary["formal_six_stage_metrics"][method]
    return {
        "response": value["response"]["contact_candidate_terminal_object_rmse"],
        "spearman": value["action_ranking"]["spearman"],
        "regret": value["action_ranking"]["top1_regret"],
        "endpoint": value["closed_loop_outcome"]["endpoint_error"],
        "success": value["closed_loop_outcome"]["success_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight10-root", type=Path, required=True)
    parser.add_argument("--weight0-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        w10_path = args.weight10_root / f"seed{seed}" / "summary.json"
        w0_path = args.weight0_root / f"seed{seed}" / "summary.json"
        w10_summary = json.loads(w10_path.read_text(encoding="utf-8"))
        w0_summary = json.loads(w0_path.read_text(encoding="utf-8"))
        w10, w0 = metrics(w10_summary, "selective_ipwm"), metrics(
            w0_summary, "selective_ipwm"
        )
        nominal = metrics(w0_summary, "shared_baseline")
        rows.append({
            "seed": seed,
            "weight10_vs_weight0": {
                "response_rmse_reduction_percent": reduction(w0["response"], w10["response"]),
                "spearman_delta": w10["spearman"] - w0["spearman"],
                "top1_regret_reduction_percent": reduction(w0["regret"], w10["regret"]),
                "endpoint_error_reduction_percent": reduction(w0["endpoint"], w10["endpoint"]),
                "success_gain_percentage_points": 100.0 * (w10["success"] - w0["success"]),
            },
            "weight0_vs_nominal": {
                "response_rmse_reduction_percent": reduction(
                    nominal["response"], w0["response"]
                ),
                "spearman_delta": w0["spearman"] - nominal["spearman"],
                "top1_regret_reduction_percent": reduction(nominal["regret"], w0["regret"]),
                "endpoint_error_reduction_percent": reduction(
                    nominal["endpoint"], w0["endpoint"]
                ),
                "success_gain_percentage_points": 100.0 * (
                    w0["success"] - nominal["success"]
                ),
            },
            "sources": {"weight10": str(w10_path), "weight0": str(w0_path)},
        })
    names = tuple(rows[0]["weight10_vs_weight0"])
    aggregate = {}
    for comparison in ("weight10_vs_weight0", "weight0_vs_nominal"):
        aggregate[comparison] = {
            name: {
                "mean": float(np.mean([row[comparison][name] for row in rows])),
                "minimum": float(np.min([row[comparison][name] for row in rows])),
                "maximum": float(np.max([row[comparison][name] for row in rows])),
                "positive_seeds": int(sum(row[comparison][name] > 0 for row in rows)),
                "total_seeds": len(rows),
            }
            for name in names
        }
    decision = aggregate["weight10_vs_weight0"]
    result = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "rows": rows,
        "aggregate": aggregate,
        "verdict": (
            "DECISION_LOSS_REGRET_ENDPOINT_DIRECTIONAL_ONLY_RESPONSE_SUCCESS_NO_GO"
            if decision["top1_regret_reduction_percent"]["positive_seeds"] >= 2
            and decision["endpoint_error_reduction_percent"]["positive_seeds"] >= 2
            and decision["success_gain_percentage_points"]["mean"] < 0.0
            and decision["response_rmse_reduction_percent"]["positive_seeds"] == 0
            else "DECISION_LOSS_ATTRIBUTION_NO_GO"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
