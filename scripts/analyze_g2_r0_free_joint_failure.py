"""Diagnose the disclosed seed-7 free-joint gate miss from frozen raw rows.

This script never changes models, thresholds, or frozen audit artifacts.  It
aggregates paired terminal-window squared errors already stored by the ICRA
audit and reports where the candidate/base difference accumulates.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = "shared_matched_adapter"
CANDIDATE = "bt_matched_adapter"


def paired_bootstrap(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default="D3__mixed_unseen")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    selected = [
        row for row in payload["rows"]
        if row["domain"] == args.domain and row["horizon"] == args.horizon
        and row["method"] in {BASE, CANDIDATE}
    ]
    by_key: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in selected:
        by_key[(row["trajectory_index"], row["window_start"])][row["method"]] = row
    if not by_key or any(set(pair) != {BASE, CANDIDATE} for pair in by_key.values()):
        raise RuntimeError("paired raw rows are missing or incomplete")

    per_window: dict[int, list[dict[str, float]]] = defaultdict(list)
    per_trajectory: dict[int, list[dict[str, float]]] = defaultdict(list)
    for (trajectory, window), pair in sorted(by_key.items()):
        base_mse = float(pair[BASE]["free_squared_error"])
        candidate_mse = float(pair[CANDIDATE]["free_squared_error"])
        item = {
            "base_mse": base_mse,
            "candidate_mse": candidate_mse,
            "mse_delta": candidate_mse - base_mse,
            "base_object_mse": float(pair[BASE]["object_squared_error"]),
            "candidate_object_mse": float(pair[CANDIDATE]["object_squared_error"]),
            "base_pusher_mse": float(pair[BASE]["pusher_xy_squared_error"]),
            "candidate_pusher_mse": float(pair[CANDIDATE]["pusher_xy_squared_error"]),
        }
        per_window[window].append(item)
        per_trajectory[trajectory].append(item)

    def summarize(items: list[dict[str, float]]) -> dict[str, float]:
        base = np.asarray([x["base_mse"] for x in items], dtype=float)
        candidate = np.asarray([x["candidate_mse"] for x in items], dtype=float)
        delta = candidate - base
        base_object = np.asarray([x["base_object_mse"] for x in items], dtype=float)
        candidate_object = np.asarray(
            [x["candidate_object_mse"] for x in items], dtype=float)
        base_pusher = np.asarray([x["base_pusher_mse"] for x in items], dtype=float)
        candidate_pusher = np.asarray(
            [x["candidate_pusher_mse"] for x in items], dtype=float)
        base_rmse = float(np.sqrt(base.mean()))
        candidate_rmse = float(np.sqrt(candidate.mean()))
        base_object_rmse = float(np.sqrt(base_object.mean()))
        candidate_object_rmse = float(np.sqrt(candidate_object.mean()))
        base_pusher_rmse = float(np.sqrt(base_pusher.mean()))
        candidate_pusher_rmse = float(np.sqrt(candidate_pusher.mean()))
        return {
            "n": int(len(items)),
            "base_rmse": base_rmse,
            "candidate_rmse": candidate_rmse,
            "relative_rmse_change_pct": 100.0 * (candidate_rmse - base_rmse) / base_rmse,
            "mean_paired_mse_delta": float(delta.mean()),
            "positive_delta_fraction": float(np.mean(delta > 0.0)),
            "object_improvement_pct": 100.0 * (
                base_object_rmse - candidate_object_rmse) / base_object_rmse,
            "pusher_change_pct": 100.0 * (
                candidate_pusher_rmse - base_pusher_rmse) / base_pusher_rmse,
        }

    trajectory_deltas = np.asarray([
        np.mean([x["mse_delta"] for x in items])
        for _, items in sorted(per_trajectory.items())
    ])
    output = {
        "version": "g2_r0_free_joint_failure_diagnosis_v1",
        "source": str(args.input.as_posix()),
        "source_version": payload.get("version"),
        "seed": payload.get("seed"),
        "domain": args.domain,
        "horizon": args.horizon,
        "methods": {"baseline": BASE, "candidate": CANDIDATE},
        "overall": summarize([x for items in per_window.values() for x in items]),
        "trajectory_cluster_mse_delta_ci95": paired_bootstrap(
            trajectory_deltas, draws=args.draws, seed=args.bootstrap_seed
        ),
        "by_window_start": [
            {"window_start": window, **summarize(items)}
            for window, items in sorted(per_window.items())
        ],
        "trajectory_mean_mse_delta": [
            {"trajectory_index": trajectory, "mean_mse_delta": float(np.mean([
                x["mse_delta"] for x in items
            ]))}
            for trajectory, items in sorted(per_trajectory.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": output["overall"],
        "trajectory_cluster_mse_delta_ci95": output["trajectory_cluster_mse_delta_ci95"],
        "by_window_start": output["by_window_start"],
    }, indent=2))


if __name__ == "__main__":
    main()
