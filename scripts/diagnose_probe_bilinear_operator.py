"""Closed-form screen for probe-conditioned bilinear action response.

The test fixes robot, physics profile, prefix and lock within each ranking set.
It compares additive history with the necessary candidate-action/history
interaction, using held-out prefixes, middle locks, and mixed physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_active_probe_identifiability import current_features, ridge_fit_predict


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def history_summary(data: dict[str, np.ndarray], *, reverse_time: bool = False) -> np.ndarray:
    action = data["probe_action"]
    joint = data["probe_joint_delta"].reshape(len(action), action.shape[1], -1)
    obj = data["probe_object_delta"][:, :, :2]
    contact = data["probe_contact"]
    if reverse_time:
        action, joint, obj, contact = action[:, ::-1], joint[:, ::-1], obj[:, ::-1], contact[:, ::-1]
    time = np.linspace(-1.0, 1.0, action.shape[1])[None, :, None]
    def summarize(value: np.ndarray) -> list[np.ndarray]:
        return [value.mean(1), value.std(1), value[:, -1] - value[:, 0], (value * time).mean(1)]
    return np.concatenate([*summarize(action), *summarize(joint), *summarize(obj),
                           contact.mean(1), contact[:, -1] - contact[:, 0]], axis=1).astype(np.float64)


def candidate_features(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([data["action"], data["ee_action_delta"], data["ee_projected_action"]], axis=1).astype(np.float64)


def interaction(summary: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.einsum("bi,bj->bij", summary, candidate).reshape(len(summary), -1)


def action_metrics(actual: np.ndarray, predicted: np.ndarray, data: dict[str, np.ndarray], select: np.ndarray) -> dict:
    robots, profiles = data["robot"].astype(str), data["profile"].astype(str)
    keys = np.asarray([f"{r}|{p}|{int(i)}|{int(l)}" for r, p, i, l in
                       zip(robots, profiles, data["prefix_id"], data["lock_index"])])
    directions = np.asarray(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))
    correlations, regrets = [], []
    for key in np.unique(keys[select]):
        rows = np.flatnonzero(select & (keys == key))
        for direction in directions:
            truth, estimate = actual[rows, :2] @ direction, predicted[rows, :2] @ direction
            correlations.append(float(np.corrcoef(rankdata(truth), rankdata(estimate))[0, 1]))
            scale = max(float(np.ptp(truth)), 1e-8)
            regrets.append(float((np.max(truth) - truth[int(np.argmax(estimate))]) / scale))
    return {"mean_spearman": float(np.mean(correlations)),
            "normalized_top1_regret": float(np.mean(regrets)), "groups": len(correlations)}


def evaluate(dataset: Path, seed: int, alpha: float) -> dict:
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots, profiles = data["robot"].astype(str), data["profile"].astype(str)
    rng = np.random.default_rng(seed)
    train_ids = {}
    for robot in ("genkiarm", "panda"):
        ids = rng.permutation(np.unique(data["prefix_id"][robots == robot]))
        train_ids[robot] = set(map(int, ids[:int(0.7 * len(ids))]))
    prefix_train = np.asarray([int(prefix) in train_ids[robot] for prefix, robot in zip(data["prefix_id"], robots)])
    middle = np.where(robots == "genkiarm", 2, 3)
    train = prefix_train & (data["lock_index"] != middle) & (profiles != "heldout_mixed")
    test = ~prefix_train & (data["lock_index"] == middle) & (profiles == "heldout_mixed")
    current, summary, reversed_summary = current_features(data), history_summary(data), history_summary(data, reverse_time=True)
    candidate = candidate_features(data)
    target = data["locked_object_step"].astype(np.float64)
    feature_sets = {
        "current": current,
        "additive_probe": np.concatenate([current, summary], axis=1),
        "bilinear_probe": np.concatenate([current, summary, interaction(summary, candidate)], axis=1),
        "bilinear_reversed_probe": np.concatenate([current, reversed_summary, interaction(reversed_summary, candidate)], axis=1),
    }
    result = {}
    for robot in ("genkiarm", "panda"):
        robot_train, robot_test = train & (robots == robot), test & (robots == robot)
        result[robot] = {}
        for name, features in feature_sets.items():
            prediction = ridge_fit_predict(features, target, robot_train, robot_test, alpha=alpha)
            rmse = float(np.sqrt(np.mean((target[robot_test] - prediction) ** 2)))
            full_prediction = np.zeros_like(target)
            full_prediction[robot_test] = prediction
            result[robot][name] = {"response_rmse": rmse, **action_metrics(target, full_prediction, data, robot_test)}
    return {"diagnostic": "probe_bilinear_operator_v1", "development_only": True,
            "group_contract": ["robot", "profile", "prefix_id", "lock_index"],
            "seed": seed, "alpha": alpha, "methods": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.seed, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
