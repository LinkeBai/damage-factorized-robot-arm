"""Aggregate strict three-seed six-stage development evidence without cherry-picking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relative_reduction(base: float, candidate: float) -> float:
    return 100.0 * (base - candidate) / base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        path = args.run_root / f"seed{seed}" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["formal_six_stage_metrics"]
        shared = metrics["shared_baseline"]
        carrier = metrics["carrier_no_intervention"]
        full = metrics["full_state_ipwm"]
        selective = metrics["selective_ipwm"]
        identical = all(
            full[section][metric] == selective[section][metric]
            for section, metric in (
                ("response", "contact_candidate_terminal_object_rmse"),
                ("action_ranking", "spearman"),
                ("action_ranking", "top1_regret"),
                ("closed_loop_outcome", "endpoint_error"),
                ("closed_loop_outcome", "success_rate"),
            )
        )
        row = {
            "seed": seed,
            "response_rmse_reduction_percent": relative_reduction(
                carrier["response"]["contact_candidate_terminal_object_rmse"],
                selective["response"]["contact_candidate_terminal_object_rmse"],
            ),
            "spearman_delta": (
                selective["action_ranking"]["spearman"]
                - carrier["action_ranking"]["spearman"]
            ),
            "top1_regret_reduction_percent": relative_reduction(
                carrier["action_ranking"]["top1_regret"],
                selective["action_ranking"]["top1_regret"],
            ),
            "endpoint_error_reduction_percent": relative_reduction(
                carrier["closed_loop_outcome"]["endpoint_error"],
                selective["closed_loop_outcome"]["endpoint_error"],
            ),
            "success_gain_percentage_points": 100.0 * (
                selective["closed_loop_outcome"]["success_rate"]
                - carrier["closed_loop_outcome"]["success_rate"]
            ),
            "contact_gain_percentage_points": 100.0 * (
                selective["contact"]["selected_candidate_rate"]
                - carrier["contact"]["selected_candidate_rate"]
            ),
            "locked_position_violation_max": selective["constraint"][
                "locked_position_violation_max"
            ],
            "locked_velocity_violation_max": selective["constraint"][
                "locked_velocity_violation_max"
            ],
            "full_state_equals_selective": identical,
            "selective_wall_time_seconds": selective["evaluation_wall_time_seconds"],
            "source": str(path),
        }
        row["versus_nominal_shared"] = {
            "response_rmse_reduction_percent": relative_reduction(
                shared["response"]["contact_candidate_terminal_object_rmse"],
                selective["response"]["contact_candidate_terminal_object_rmse"],
            ),
            "spearman_delta": (
                selective["action_ranking"]["spearman"]
                - shared["action_ranking"]["spearman"]
            ),
            "top1_regret_reduction_percent": relative_reduction(
                shared["action_ranking"]["top1_regret"],
                selective["action_ranking"]["top1_regret"],
            ),
            "endpoint_error_reduction_percent": relative_reduction(
                shared["closed_loop_outcome"]["endpoint_error"],
                selective["closed_loop_outcome"]["endpoint_error"],
            ),
            "success_gain_percentage_points": 100.0 * (
                selective["closed_loop_outcome"]["success_rate"]
                - shared["closed_loop_outcome"]["success_rate"]
            ),
        }
        rows.append(row)
    metric_names = [
        "response_rmse_reduction_percent",
        "spearman_delta",
        "top1_regret_reduction_percent",
        "endpoint_error_reduction_percent",
        "success_gain_percentage_points",
        "contact_gain_percentage_points",
    ]
    aggregate = {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "minimum": float(np.min([row[name] for row in rows])),
            "maximum": float(np.max([row[name] for row in rows])),
            "positive_seeds": int(sum(row[name] > 0 for row in rows)),
            "total_seeds": len(rows),
        }
        for name in metric_names
    }
    nominal_aggregate = {
        name: {
            "mean": float(np.mean([
                row["versus_nominal_shared"][name] for row in rows
            ])),
            "minimum": float(np.min([
                row["versus_nominal_shared"][name] for row in rows
            ])),
            "maximum": float(np.max([
                row["versus_nominal_shared"][name] for row in rows
            ])),
            "positive_seeds": int(sum(
                row["versus_nominal_shared"][name] > 0 for row in rows
            )),
            "total_seeds": len(rows),
        }
        for name in (
            "response_rmse_reduction_percent",
            "spearman_delta",
            "top1_regret_reduction_percent",
            "endpoint_error_reduction_percent",
            "success_gain_percentage_points",
        )
    }
    gate = {
        "spearman_delta_at_least_0_10_each_passing_seed": int(sum(
            row["spearman_delta"] >= 0.10 for row in rows
        )),
        "endpoint_reduction_at_least_10_percent_each_passing_seed": int(sum(
            row["endpoint_error_reduction_percent"] >= 10.0 for row in rows
        )),
        "direction_consistency_2_of_3": (
            aggregate["spearman_delta"]["positive_seeds"] >= 2
            and aggregate["endpoint_error_reduction_percent"]["positive_seeds"] >= 2
        ),
        "magnitude_gate_passed": (
            sum(row["spearman_delta"] >= 0.10 for row in rows) >= 2
            and sum(row["endpoint_error_reduction_percent"] >= 10.0 for row in rows) >= 2
        ),
        "zero_constraint_violation_all_seeds": all(
            row["locked_position_violation_max"] == 0.0
            and row["locked_velocity_violation_max"] == 0.0
            for row in rows
        ),
        "path_support_has_distinct_effect": not all(
            row["full_state_equals_selective"] for row in rows
        ),
    }
    result = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "role": "strict_D2_D4_development_evidence",
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
        "versus_nominal_shared_aggregate": nominal_aggregate,
        "gate": gate,
        "verdict": (
            "PASS" if gate["magnitude_gate_passed"]
            and gate["zero_constraint_violation_all_seeds"]
            and gate["path_support_has_distinct_effect"]
            else "DIRECTIONAL_SIGNAL_MAGNITUDE_AND_ATTRIBUTION_NO_GO"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
