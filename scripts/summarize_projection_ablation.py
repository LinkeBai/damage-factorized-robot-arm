"""Aggregate the three-seed analytic-projection removal ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "nonzero_seeds": int(sum(value > 0 for value in values)),
        "total_seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-projection-root", type=Path, required=True)
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        no_path = args.no_projection_root / f"seed{seed}" / "summary.json"
        yes_path = args.projected_root / f"seed{seed}" / "summary.json"
        no_summary = json.loads(no_path.read_text(encoding="utf-8"))
        yes_summary = json.loads(yes_path.read_text(encoding="utf-8"))
        no_projection = no_summary["formal_six_stage_metrics"]["bt_dpwm"]
        projected = yes_summary["formal_six_stage_metrics"]["selective_ipwm"]
        rows.append({
            "seed": seed,
            "without_projection": {
                "locked_position_violation_max_rad": no_projection["constraint"][
                    "locked_position_violation_max"
                ],
                "locked_velocity_violation_max_rad_s": no_projection["constraint"][
                    "locked_velocity_violation_max"
                ],
                "top1_regret": no_projection["action_ranking"]["top1_regret"],
                "endpoint_error": no_projection["closed_loop_outcome"]["endpoint_error"],
                "success_rate": no_projection["closed_loop_outcome"]["success_rate"],
            },
            "with_projection": {
                "locked_position_violation_max_rad": projected["constraint"][
                    "locked_position_violation_max"
                ],
                "locked_velocity_violation_max_rad_s": projected["constraint"][
                    "locked_velocity_violation_max"
                ],
                "top1_regret": projected["action_ranking"]["top1_regret"],
                "endpoint_error": projected["closed_loop_outcome"]["endpoint_error"],
                "success_rate": projected["closed_loop_outcome"]["success_rate"],
            },
            "sources": {"without_projection": str(no_path), "with_projection": str(yes_path)},
        })
    pos = [row["without_projection"]["locked_position_violation_max_rad"] for row in rows]
    vel = [row["without_projection"]["locked_velocity_violation_max_rad_s"] for row in rows]
    projected_zero = all(
        row["with_projection"]["locked_position_violation_max_rad"] == 0.0
        and row["with_projection"]["locked_velocity_violation_max_rad_s"] == 0.0
        for row in rows
    )
    result = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "role": "analytic_projection_ablation",
        "seeds": seeds,
        "rows": rows,
        "without_projection_position_violation_rad": stats(pos),
        "without_projection_position_violation_degrees": stats(
            [value * 180.0 / np.pi for value in pos]
        ),
        "without_projection_velocity_violation_rad_s": stats(vel),
        "with_projection_zero_violation_all_seeds": projected_zero,
        "verdict": (
            "ANALYTIC_PROJECTION_CONSTRAINT_GO"
            if projected_zero and all(value > 0 for value in pos) and all(value > 0 for value in vel)
            else "ANALYTIC_PROJECTION_CONSTRAINT_NO_GO"
        ),
        "scope_note": (
            "This ablation establishes exact structural constraint satisfaction; "
            "it does not by itself establish better task success."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
