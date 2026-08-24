"""Summarize enlarged frozen-checkpoint evaluation with paired cluster bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ("D3__mixed_composition", "D3__mixed_unseen")
HORIZONS = (10, 25, 50)
BASELINE = "shared_matched_adapter"
METHODS = (
    "bt_matched_adapter",
    "bt_no_geometry_matched_adapter",
    "bt_no_latent_matched_adapter",
    "bt_no_intervention_matched_adapter",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clustered_improvement(rows, domain, horizon, method, rng, draws):
    selected = [row for row in rows
                if row["domain"] == domain and row["horizon"] == horizon
                and row["method"] in (BASELINE, method)]
    by_key = {(row["trajectory_index"], row["window_start"], row["method"]):
              row["object_squared_error"] for row in selected}
    trajectories = sorted({key[0] for key in by_key})
    windows = sorted({key[1] for key in by_key})
    base = np.asarray([[by_key[(trajectory, window, BASELINE)] for window in windows]
                       for trajectory in trajectories], dtype=np.float64)
    candidate = np.asarray([[by_key[(trajectory, window, method)] for window in windows]
                            for trajectory in trajectories], dtype=np.float64)
    point = 100.0 * (np.sqrt(base.mean()) - np.sqrt(candidate.mean())) / np.sqrt(base.mean())
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen = rng.integers(0, len(trajectories), len(trajectories))
        base_rmse = np.sqrt(base[chosen].mean())
        method_rmse = np.sqrt(candidate[chosen].mean())
        samples[index] = 100.0 * (base_rmse - method_rmse) / base_rmse
    return {
        "object_improvement_pct": float(point),
        "cluster_bootstrap_95ci_pct": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "trajectory_clusters": len(trajectories),
        "terminal_windows_per_trajectory": len(windows),
        "paired_terminal_samples": int(base.size),
        "probability_improvement_positive": float((samples > 0).mean()),
        "baseline_object_rmse": float(np.sqrt(base.mean())),
        "method_object_rmse": float(np.sqrt(candidate.mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20260824)
    result = {
        "version": "g2_r0_icra_audit_summary_v1",
        "bootstrap": "paired trajectory-cluster bootstrap",
        "bootstrap_draws": args.draws,
        "bootstrap_seed": 20260824,
        "seeds": {},
        "artifacts": [],
    }
    all_full = []
    for path in (Path("scripts/evaluate_g2_r0_core_metrics.py"),
                 Path("scripts/summarize_g2_r0_icra_audit.py")):
        result["artifacts"].append({
            "path": path.as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    for seed in args.seeds:
        raw_path = args.root / f"seed{seed}" / "raw_window_metrics_30traj.json"
        metrics_path = args.root / f"seed{seed}" / "metrics_30traj.json"
        rows = json.loads(raw_path.read_text(encoding="utf-8"))["rows"]
        cells = []
        for domain in DOMAINS:
            for horizon in HORIZONS:
                for method in METHODS:
                    cell = {"domain": domain, "horizon": horizon, "method": method,
                            **clustered_improvement(rows, domain, horizon, method,
                                                    rng, args.draws)}
                    cells.append(cell)
                    if method == "bt_matched_adapter":
                        all_full.append(cell)
        result["seeds"][str(seed)] = {"cells": cells}
        for path in (raw_path, metrics_path):
            result["artifacts"].append({
                "path": path.as_posix(), "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
        run_names = {7: "seed7_centered_regression_v1",
                     17: "seed17_centered_v3",
                     27: "seed27_confirmation_v1"}
        config_names = {
            7: "g2_r0_physical_context_residual_seed7_regression_v1.yaml",
            17: "g2_r0_physical_context_residual_dev_v2.yaml",
            27: "g2_r0_physical_context_residual_seed27_confirmation_v1.yaml",
            37: "g2_r0_physical_context_residual_extension_v1.yaml",
            47: "g2_r0_physical_context_residual_extension_v1.yaml",
        }
        model_path = ((Path("runs/g2_r0_physical_context_residual") /
                       run_names[seed] / "model.pt") if seed in run_names else
                      (Path("runs/g2_r0_physical_context_residual_extension") /
                       f"seed{seed}_v1/model.pt"))
        dependencies = (
            model_path,
            Path(f"runs/g2_bt_dpwm_meta_train_z32/seed{seed}_v1/baseline_model.pt"),
            Path(f"runs/g2_bt_dpwm_context_encoder_z65/seed{seed}_v1/context_encoder.pt"),
            Path(f"runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_v1/shared_adapter.pt"),
            Path(f"runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_v1/bt_adapter.pt"),
            Path("config/experiment") / config_names[seed],
        )
        for path in dependencies:
            result["artifacts"].append({
                "path": path.as_posix(), "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    result["gate"] = {
        "full_model_all_point_estimates_positive": all(
            row["object_improvement_pct"] > 0 for row in all_full),
        "full_model_all_ci_lower_bounds_positive": all(
            row["cluster_bootstrap_95ci_pct"][0] > 0 for row in all_full),
        "minimum_full_model_point_improvement_pct": min(
            row["object_improvement_pct"] for row in all_full),
        "minimum_full_model_ci_lower_bound_pct": min(
            row["cluster_bootstrap_95ci_pct"][0] for row in all_full),
    }
    cross_seed = []
    for domain in DOMAINS:
        for horizon in HORIZONS:
            values = np.asarray([
                row["object_improvement_pct"] for row in all_full
                if row["domain"] == domain and row["horizon"] == horizon
            ], dtype=np.float64)
            samples = np.asarray([
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(args.draws)
            ])
            cross_seed.append({
                "domain": domain,
                "horizon": horizon,
                "seed_count": len(values),
                "mean_object_improvement_pct": float(values.mean()),
                "seed_bootstrap_95ci_pct": [
                    float(x) for x in np.quantile(samples, [0.025, 0.975])
                ],
                "per_seed_object_improvement_pct": values.tolist(),
            })
    result["cross_seed_summary"] = cross_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
