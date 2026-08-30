"""Audit within-context action ranking from a contact-probe secant operator.

Unlike the earlier diagnostic, every ranking group fixes robot, physics
profile, prefix, and lock.  The probe may therefore help only by changing the
candidate-action response, not by identifying which physics profile a row came
from.  This is a development-only information test on the existing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def secant_response(probe_action: np.ndarray, probe_delta: np.ndarray, ridge: float) -> np.ndarray:
    action = probe_action - probe_action.mean(axis=0, keepdims=True)
    effect = probe_delta[:, :2] - probe_delta[:, :2].mean(axis=0, keepdims=True)
    gram = action.T @ action + ridge * np.eye(action.shape[1])
    return np.linalg.solve(gram, action.T @ effect).T


def evaluate(dataset: Path, seed: int, ridge: float) -> dict:
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str)
    profiles = data["profile"].astype(str)
    rng = np.random.default_rng(seed)
    test_prefix = {}
    for robot in ("genkiarm", "panda"):
        ids = rng.permutation(np.unique(data["prefix_id"][robots == robot]))
        test_prefix[robot] = set(map(int, ids[int(0.7 * len(ids)) :]))
    middle = np.where(robots == "genkiarm", 2, 3)
    test = np.asarray([int(prefix) in test_prefix[robot] for prefix, robot in zip(data["prefix_id"], robots)])
    test &= data["lock_index"] == middle
    test &= profiles == "heldout_mixed"

    keys = np.asarray([
        f"{robot}|{profile}|{int(prefix)}|{int(lock)}"
        for robot, profile, prefix, lock in zip(robots, profiles, data["prefix_id"], data["lock_index"])
    ])
    groups = [np.flatnonzero(test & (keys == key)) for key in np.unique(keys[test])]
    group_operators = []
    for rows in groups:
        row = rows[0]
        group_operators.append(secant_response(data["probe_action"][row], data["probe_object_delta"][row], ridge))
    permutation = rng.permutation(len(groups))
    directions = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    methods = {name: {"correlation": [], "regret": [], "by_robot": {"genkiarm": [], "panda": []}}
               for name in ("ee_projected", "ordered_secant", "permuted_secant")}

    for group_index, rows in enumerate(groups):
        robot = robots[rows[0]]
        action = data["action"][rows]
        center = data["probe_action"][rows[0]].mean(axis=0, keepdims=True)
        predictions = {
            "ee_projected": data["ee_projected_action"][rows, :2],
            "ordered_secant": (action - center) @ group_operators[group_index].T,
            "permuted_secant": (action - center) @ group_operators[permutation[group_index]].T,
        }
        truth = data["locked_object_step"][rows, :2]
        for direction in directions:
            target = truth @ direction
            for name, prediction in predictions.items():
                estimate = prediction @ direction
                correlation = float(np.corrcoef(rankdata(target), rankdata(estimate))[0, 1])
                scale = max(float(np.ptp(target)), 1e-8)
                regret = float((np.max(target) - target[int(np.argmax(estimate))]) / scale)
                methods[name]["correlation"].append(correlation)
                methods[name]["regret"].append(regret)
                methods[name]["by_robot"][robot].append((correlation, regret))

    rendered = {}
    for name, values in methods.items():
        rendered[name] = {
            "mean_spearman": float(np.mean(values["correlation"])),
            "normalized_top1_regret": float(np.mean(values["regret"])),
            "groups": len(values["correlation"]),
            "by_robot": {
                robot: {
                    "mean_spearman": float(np.mean([item[0] for item in rows])),
                    "normalized_top1_regret": float(np.mean([item[1] for item in rows])),
                }
                for robot, rows in values["by_robot"].items()
            },
        }
    return {
        "diagnostic": "probe_conditioned_secant_operator_v1",
        "development_only": True,
        "group_contract": ["robot", "profile", "prefix_id", "lock_index"],
        "test_contract": "heldout prefix + heldout middle lock + heldout mixed physics",
        "seed": seed,
        "ridge": ridge,
        "methods": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.seed, args.ridge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
