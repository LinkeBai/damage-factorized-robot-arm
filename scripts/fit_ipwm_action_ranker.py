"""Fit a development-only local action-cost ranker from branch-rollout rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_ipwm_action_ranking import spearman


FEATURES = (
    "carrier_predicted_cost_m",
    "selective_predicted_cost_m",
    "predicted_cost_delta_m",
    "action_deviation_rms",
    "first_action_deviation_l2",
    "action_effort_rms",
)


def grouped(rows):
    result = {}
    for row in rows:
        result.setdefault(row["target"], []).append(row)
    return result


def centered_matrix(rows):
    blocks_x, blocks_y, target_ids = [], [], []
    for target, group in grouped(rows).items():
        x = np.asarray([[float(row[key]) for key in FEATURES] for row in group])
        y = np.asarray([float(row["true_cost_m"]) for row in group])
        blocks_x.append(x - x.mean(axis=0, keepdims=True))
        blocks_y.append(y - y.mean())
        target_ids.extend([target] * len(group))
    return np.vstack(blocks_x), np.concatenate(blocks_y), target_ids


def fit_ridge(rows, alpha):
    x, y, _ = centered_matrix(rows)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = x / scale
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ y)
    return coef, scale


def evaluate(rows, coef, scale):
    metrics = []
    for target, group in grouped(rows).items():
        x = np.asarray([[float(row[key]) for key in FEATURES] for row in group])
        y = np.asarray([float(row["true_cost_m"]) for row in group])
        score = ((x - x.mean(axis=0, keepdims=True)) / scale) @ coef
        chosen, oracle = int(np.argmin(score)), int(np.argmin(y))
        carrier_cost = x[:, FEATURES.index("carrier_predicted_cost_m")]
        safe = np.argsort(carrier_cost)[: max(2, len(group) // 4)]
        safe_chosen = int(safe[np.argmin(score[safe])])
        metrics.append({
            "target": target, "spearman": spearman(score, y),
            "chosen_index": int(group[chosen]["candidate_index"]),
            "oracle_index": int(group[oracle]["candidate_index"]),
            "chosen_realized_cost_m": float(y[chosen]),
            "oracle_realized_cost_m": float(y[oracle]),
            "regret_m": float(y[chosen] - y[oracle]),
            "carrier_safe_chosen_index": int(group[safe_chosen]["candidate_index"]),
            "carrier_safe_chosen_realized_cost_m": float(y[safe_chosen]),
            "carrier_safe_regret_m": float(y[safe_chosen] - y[oracle]),
        })
    return metrics


def load_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))["candidate_rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train, validation = load_rows(args.train), load_rows(args.validation)
    candidates = []
    for alpha in (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
        coef, scale = fit_ridge(train, alpha)
        metrics = evaluate(validation, coef, scale)
        candidates.append({
            "alpha": alpha, "coef": coef, "scale": scale, "metrics": metrics,
            "mean_spearman": float(np.mean([m["spearman"] for m in metrics])),
            "mean_regret_m": float(np.mean([m["regret_m"] for m in metrics])),
        })
    best = max(candidates, key=lambda item: (item["mean_spearman"], -item["mean_regret_m"]))
    payload = {
        "version": "ipwm_local_action_ranker_v1", "development_only": True,
        "features": list(FEATURES), "feature_centering": "within_candidate_set",
        "alpha": best["alpha"], "coefficient": best["coef"].tolist(),
        "feature_scale": best["scale"].tolist(),
        "training_metrics": evaluate(train, best["coef"], best["scale"]),
        "validation_metrics": best["metrics"],
        "validation_mean_spearman": best["mean_spearman"],
        "validation_mean_regret_m": best["mean_regret_m"],
        "alpha_search": [{
            "alpha": item["alpha"], "mean_spearman": item["mean_spearman"],
            "mean_regret_m": item["mean_regret_m"],
        } for item in candidates],
    }
    if args.test is not None:
        test_metrics = evaluate(load_rows(args.test), best["coef"], best["scale"])
        payload["test_metrics"] = test_metrics
        payload["test_mean_spearman"] = float(np.mean([m["spearman"] for m in test_metrics]))
        payload["test_mean_regret_m"] = float(np.mean([m["regret_m"] for m in test_metrics]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
