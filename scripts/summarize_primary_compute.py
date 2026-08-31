"""Record available formal-evaluation compute without inventing missing training time."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipwm-root", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        ipwm_path = args.ipwm_root / f"seed{seed}" / "summary.json"
        global_path = args.global_root / f"seed{seed}" / "summary.json"
        ipwm = json.loads(ipwm_path.read_text(encoding="utf-8"))
        global_summary = json.loads(global_path.read_text(encoding="utf-8"))
        ipwm_metrics = ipwm["formal_six_stage_metrics"]
        global_metrics = global_summary["formal_six_stage_metrics"]
        rows.append({
            "seed": seed,
            "groups": ipwm_metrics["selective_ipwm"]["groups"],
            "candidates_per_group": 128,
            "horizon_steps": 50,
            "ipwm_parameters": ipwm["parameters"],
            "global_matched_parameters": global_summary["parameters"],
            "evaluation_wall_time_seconds": {
                "nominal": ipwm_metrics["shared_baseline"]["evaluation_wall_time_seconds"],
                "carrier": ipwm_metrics["carrier_no_intervention"]["evaluation_wall_time_seconds"],
                "full_state_ipwm": ipwm_metrics["full_state_ipwm"]["evaluation_wall_time_seconds"],
                "selective_ipwm": ipwm_metrics["selective_ipwm"]["evaluation_wall_time_seconds"],
                "global_matched": global_metrics["projection_global_residual_matched"][
                    "evaluation_wall_time_seconds"
                ],
            },
            "recorded_device_ipwm": ipwm["device"],
            "recorded_device_global": global_summary["device"],
            "sources": {"ipwm": str(ipwm_path), "global": str(global_path)},
        })
    method_names = tuple(rows[0]["evaluation_wall_time_seconds"])
    aggregate = {
        name: {
            "mean_seconds": float(np.mean([
                row["evaluation_wall_time_seconds"][name] for row in rows
            ])),
            "minimum_seconds": float(np.min([
                row["evaluation_wall_time_seconds"][name] for row in rows
            ])),
            "maximum_seconds": float(np.max([
                row["evaluation_wall_time_seconds"][name] for row in rows
            ])),
        }
        for name in method_names
    }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    result = {
        "protocol": "icra_2027_primary_5dof_recovery_v1",
        "scope": "formal 400x128x50 six-stage evaluation",
        "rows": rows,
        "aggregate_evaluation_wall_time": aggregate,
        "runtime_used_to_generate_this_ledger": runtime,
        "training_wall_time": None,
        "training_wall_time_note": (
            "Historical training summaries did not reliably persist end-to-end wall time; "
            "it is reported as missing rather than reconstructed."
        ),
        "timing_limitations": (
            "IPWM and global rows were produced on different recorded devices in some seeds; "
            "wall time documents resource cost and is not used for a speed-comparison claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
