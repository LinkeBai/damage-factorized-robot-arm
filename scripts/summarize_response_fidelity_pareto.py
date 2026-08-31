"""Summarize the frozen response-fidelity versus decision-loss development study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


METHOD = "projection_global_residual_matched"
BASELINE = "shared_baseline"


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["formal_six_stage_metrics"]


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def row(seed: int, path: Path) -> dict:
    metrics = load_metrics(path)
    baseline = metrics[BASELINE]
    candidate = metrics[METHOD]
    response_base = baseline["response"]["contact_candidate_terminal_object_rmse"]
    response_model = candidate["response"]["contact_candidate_terminal_object_rmse"]
    regret_base = baseline["action_ranking"]["top1_regret"]
    regret_model = candidate["action_ranking"]["top1_regret"]
    endpoint_base = baseline["closed_loop_outcome"]["endpoint_error"]
    endpoint_model = candidate["closed_loop_outcome"]["endpoint_error"]
    return {
        "seed": seed,
        "source": path.as_posix(),
        "response_rmse_improvement_pct": improvement(response_base, response_model),
        "top1_regret_improvement_pct": improvement(regret_base, regret_model),
        "endpoint_error_improvement_pct": improvement(endpoint_base, endpoint_model),
        "success_delta_pp": 100.0 * (
            candidate["closed_loop_outcome"]["success_rate"]
            - baseline["closed_loop_outcome"]["success_rate"]
        ),
        "spearman_delta": (
            candidate["action_ranking"]["spearman"]
            - baseline["action_ranking"]["spearman"]
        ),
        "locked_position_violation_max": candidate["constraint"]["locked_position_violation_max"],
        "locked_velocity_violation_max": candidate["constraint"]["locked_velocity_violation_max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight0-root", type=Path, required=True)
    parser.add_argument("--weight003-seed27", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row(seed, args.weight0_root / f"seed{seed}" / "summary.json")
        for seed in (7, 17, 27)
    ]
    keys = (
        "response_rmse_improvement_pct",
        "top1_regret_improvement_pct",
        "endpoint_error_improvement_pct",
        "success_delta_pp",
        "spearman_delta",
    )
    aggregate = {
        key: {
            "mean": mean(item[key] for item in rows),
            "positive_seeds": sum(item[key] > 0.0 for item in rows),
            "total_seeds": len(rows),
        }
        for key in keys
    }
    weight003 = row(27, args.weight003_seed27)
    payload = {
        "status": "DEVELOPMENT_PARETO_FROZEN",
        "scope": "D2/D4 development only; not D3 confirmation and not real robot",
        "model_identity": (
            "same-capacity global residual with analytic projection; this is not selective IPWM"
        ),
        "weight0": {"decision_weight": 0.0, "rows": rows, "aggregate": aggregate},
        "weight003_seed27": {"decision_weight": 0.03, "row": weight003},
        "selection": {
            "chosen_development_ablation": "weight0",
            "reason": (
                "weight0 improves response RMSE in 3/3 seeds; weight0.03 reverses the "
                "seed27 response gain and does not improve regret or endpoint"
            ),
            "promotion": "ablation/supporting result only",
            "not_supported": [
                "selective-IPWM attribution",
                "3/3 seed control-outcome improvement",
                "D3 confirmation improvement",
                "real-robot benefit",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
