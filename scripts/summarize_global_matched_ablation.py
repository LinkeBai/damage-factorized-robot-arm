"""Aggregate the preregistered same-capacity global-residual attribution control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reduction(base: float, value: float) -> float:
    return 100.0 * (base - value) / base


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_seeds": int(sum(value > 0 for value in values)),
        "total_seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--ipwm-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        global_path = args.global_root / f"seed{seed}" / "summary.json"
        ipwm_path = args.ipwm_root / f"seed{seed}" / "summary.json"
        global_summary = json.loads(global_path.read_text(encoding="utf-8"))
        ipwm_summary = json.loads(ipwm_path.read_text(encoding="utf-8"))
        baseline = global_summary["formal_six_stage_metrics"]["shared_baseline"]
        global_model = global_summary["formal_six_stage_metrics"][
            "projection_global_residual_matched"
        ]
        ipwm = ipwm_summary["formal_six_stage_metrics"]["selective_ipwm"]

        def metrics(method: dict) -> dict[str, float]:
            return {
                "response_rmse": method["response"][
                    "contact_candidate_terminal_object_rmse"
                ],
                "spearman": method["action_ranking"]["spearman"],
                "top1_regret": method["action_ranking"]["top1_regret"],
                "endpoint_error": method["closed_loop_outcome"]["endpoint_error"],
                "success_rate": method["closed_loop_outcome"]["success_rate"],
            }

        base_metrics = metrics(baseline)
        global_metrics = metrics(global_model)
        ipwm_metrics = metrics(ipwm)
        rows.append({
            "seed": seed,
            "baseline": base_metrics,
            "global_matched": global_metrics,
            "ipwm": ipwm_metrics,
            "global_vs_nominal": {
                "response_rmse_reduction_percent": reduction(
                    base_metrics["response_rmse"], global_metrics["response_rmse"]
                ),
                "spearman_delta": global_metrics["spearman"] - base_metrics["spearman"],
                "top1_regret_reduction_percent": reduction(
                    base_metrics["top1_regret"], global_metrics["top1_regret"]
                ),
                "endpoint_error_reduction_percent": reduction(
                    base_metrics["endpoint_error"], global_metrics["endpoint_error"]
                ),
                "success_gain_percentage_points": 100.0 * (
                    global_metrics["success_rate"] - base_metrics["success_rate"]
                ),
            },
            "ipwm_minus_global": {
                "response_rmse_reduction_percent": reduction(
                    global_metrics["response_rmse"], ipwm_metrics["response_rmse"]
                ),
                "spearman_delta": ipwm_metrics["spearman"] - global_metrics["spearman"],
                "top1_regret_reduction_percent": reduction(
                    global_metrics["top1_regret"], ipwm_metrics["top1_regret"]
                ),
                "endpoint_error_reduction_percent": reduction(
                    global_metrics["endpoint_error"], ipwm_metrics["endpoint_error"]
                ),
                "success_gain_percentage_points": 100.0 * (
                    ipwm_metrics["success_rate"] - global_metrics["success_rate"]
                ),
            },
            "sources": {"global": str(global_path), "ipwm": str(ipwm_path)},
        })

    names = (
        "response_rmse_reduction_percent",
        "spearman_delta",
        "top1_regret_reduction_percent",
        "endpoint_error_reduction_percent",
        "success_gain_percentage_points",
    )
    result = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "role": "same_capacity_global_residual_attribution_control",
        "seeds": seeds,
        "rows": rows,
        "global_vs_nominal_aggregate": {
            name: aggregate([row["global_vs_nominal"][name] for row in rows])
            for name in names
        },
        "ipwm_minus_global_aggregate": {
            name: aggregate([row["ipwm_minus_global"][name] for row in rows])
            for name in names
        },
    }
    structural = result["ipwm_minus_global_aggregate"]
    result["attribution_gate"] = {
        "ipwm_better_regret_2_of_3": structural[
            "top1_regret_reduction_percent"
        ]["positive_seeds"] >= 2,
        "ipwm_better_endpoint_2_of_3": structural[
            "endpoint_error_reduction_percent"
        ]["positive_seeds"] >= 2,
        "ipwm_better_spearman_2_of_3": structural["spearman_delta"][
            "positive_seeds"
        ] >= 2,
    }
    result["verdict"] = (
        "STRUCTURAL_ATTRIBUTION_GO"
        if all(result["attribution_gate"].values())
        else "STRUCTURAL_ATTRIBUTION_NO_GO"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
