"""Diagnose whether current-state EE action response identifies object response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize(data, rows):
    response = data["ee_projected_action"][rows].astype(np.float64)
    target = data["locked_object_step"][rows].astype(np.float64)
    robot_code = (data["robot"][rows].astype(str) == "panda").astype(np.int64)
    keys = np.stack([
        robot_code, data["prefix_id"][rows], data["lock_index"][rows]
    ], axis=1)
    within_position, within_twist = [], []
    distance_pearson, distance_spearman = [], []
    xy_ranges, best_margins = [], []
    directions = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=np.float64)
    for key in np.unique(keys, axis=0):
        local = np.flatnonzero(np.all(keys == key, axis=1))
        if len(local) < 3:
            continue
        y, x = target[local], response[local]
        within_position.append(float(np.mean(np.var(y[:, :3], axis=0))))
        within_twist.append(float(np.mean(np.var(y[:, 3:], axis=0))))
        dx, dy = [], []
        for i in range(len(local)):
            for j in range(i + 1, len(local)):
                dx.append(float(np.linalg.norm(x[i] - x[j])))
                dy.append(float(np.linalg.norm(y[i] - y[j])))
        pearson = safe_correlation(np.asarray(dx), np.asarray(dy))
        spearman = safe_correlation(rankdata(np.asarray(dx)), rankdata(np.asarray(dy)))
        if pearson is not None:
            distance_pearson.append(pearson)
            distance_spearman.append(spearman)
        for direction in directions:
            scores = y[:, :2] @ direction
            ordered = np.sort(scores)
            xy_ranges.append(float(ordered[-1] - ordered[0]))
            best_margins.append(float(ordered[-1] - ordered[-2]))
    position_total = float(np.mean(np.var(target[:, :3], axis=0)))
    twist_total = float(np.mean(np.var(target[:, 3:], axis=0)))
    position_within = float(np.mean(within_position))
    twist_within = float(np.mean(within_twist))
    return {
        "rows": int(len(rows)), "prefix_lock_groups": int(len(within_position)),
        "candidate_action_variance_fraction": {
            "position": position_within / max(position_total, 1e-16),
            "twist": twist_within / max(twist_total, 1e-16),
        },
        "ee_to_object_pairwise_distance": {
            "mean_pearson": float(np.mean(distance_pearson)),
            "mean_spearman": float(np.mean(distance_spearman)),
            "positive_spearman_group_fraction": float(np.mean(np.asarray(distance_spearman) > 0)),
        },
        "candidate_xy_effect_m": {
            "median_range": float(np.median(xy_ranges)),
            "p10_range": float(np.quantile(xy_ranges, 0.1)),
            "median_best_margin": float(np.median(best_margins)),
            "p10_best_margin": float(np.quantile(best_margins, 0.1)),
        },
    }


def run(dataset: Path):
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str)
    report = {
        "version": "contact_action_observability_diagnostic_v1",
        "dataset": str(dataset),
        "interpretation_contract": {
            "action_variance_fraction": "fraction of total response variance attributable to candidates within the same prefix and lock",
            "distance_correlation": "whether nearby analytic EE action effects imply nearby object effects",
            "not_a_model_gate": True,
        },
        "by_robot": {},
    }
    for robot in ("genkiarm", "panda"):
        report["by_robot"][robot] = summarize(data, np.flatnonzero(robots == robot))
    report["pooled"] = summarize(data, np.arange(len(robots)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
