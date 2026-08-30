"""Hierarchical paired five-seed summary for frozen GenkiArm confirmation V2."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DOMAINS = ("D3__high_damping", "D3__mixed_composition", "D3__mixed_unseen")
HORIZONS = (10, 25, 50)
METHODS = ("routed_selective_ipwm", "selective_ipwm")


def _paired(raw_paths: list[Path], method: str) -> dict[int, dict[tuple, list[tuple[float, ...]]]]:
    result: dict[int, dict[tuple, list[tuple[float, ...]]]] = defaultdict(lambda: defaultdict(list))
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8")); seed = int(payload["seed"])
        keyed = {(r["domain"], int(r["horizon"]), int(r["window_start"]),
                  int(r["trajectory_index"]), r["method"]): r for r in payload["rows"]}
        for domain in DOMAINS:
            for horizon in HORIZONS:
                identities = sorted({(k[2], k[3]) for k in keyed if k[0] == domain and k[1] == horizon})
                for window, trajectory in identities:
                    carrier = keyed[(domain, horizon, window, trajectory, "carrier_no_intervention")]
                    candidate = keyed[(domain, horizon, window, trajectory, method)]
                    result[seed][(domain, horizon)].append((
                        float(carrier["object_squared_error"]), float(candidate["object_squared_error"]),
                        float(carrier["free_squared_error"]), float(candidate["free_squared_error"]),
                        float(candidate["violation_squared_error"])))
    return result


def _effect(rows: list[tuple[float, ...]]) -> tuple[float, float]:
    values = np.asarray(rows, dtype=float)
    object_carrier, object_candidate = np.sqrt(values[:, 0].mean()), np.sqrt(values[:, 1].mean())
    free_carrier, free_candidate = np.sqrt(values[:, 2].mean()), np.sqrt(values[:, 3].mean())
    return (100 * (object_carrier-object_candidate) / object_carrier,
            100 * (free_candidate-free_carrier) / free_carrier)


def summarize(raw_paths: list[Path], draws: int = 20_000, rng_seed: int = 20260829) -> dict:
    summaries = {}; rng = np.random.default_rng(rng_seed)
    for method in METHODS:
        paired = _paired(raw_paths, method); seeds = sorted(paired)
        seed_effects = {}
        for seed in seeds:
            rows = [row for cell in paired[seed].values() for row in cell]
            seed_effects[str(seed)] = dict(zip(("object_improvement_pct", "free_regression_pct"), _effect(rows)))
        bootstrap = np.empty(draws)
        for draw in range(draws):
            effects = []
            for seed in rng.choice(seeds, size=len(seeds), replace=True):
                sampled = []
                for cell in paired[int(seed)].values():
                    indices = rng.integers(0, len(cell), size=len(cell))
                    sampled.extend(cell[index] for index in indices)
                effects.append(_effect(sampled)[0])
            bootstrap[draw] = np.mean(effects)
        objects = np.array([v["object_improvement_pct"] for v in seed_effects.values()])
        frees = np.array([v["free_regression_pct"] for v in seed_effects.values()])
        summaries[method] = {
            "seed_effects": seed_effects, "mean_object_improvement_pct": float(objects.mean()),
            "positive_seed_fraction": float(np.mean(objects > 0)),
            "hierarchical_paired_bootstrap_95_ci_pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "max_free_regression_pct": float(frees.max()),
            "max_locked_coordinate_violation_rms": float(max(
                np.sqrt(row[4]) for seed in paired.values() for cell in seed.values() for row in cell)),
        }
    primary = summaries["routed_selective_ipwm"]
    gate_checks = {
        "mean_object_improvement_at_least_5pct": primary["mean_object_improvement_pct"] >= 5.0,
        "positive_seed_fraction_at_least_0p8": primary["positive_seed_fraction"] >= 0.8,
        "bootstrap_lower_bound_above_zero": primary["hierarchical_paired_bootstrap_95_ci_pct"][0] > 0.0,
        "no_free_joint_regression": primary["max_free_regression_pct"] <= 1e-9,
        "locked_coordinate_violation_at_most_1e_7": primary["max_locked_coordinate_violation_rms"] <= 1e-7,
    }
    return {"version": "genkiarm_confirmation_v2_summary", "seeds": sorted(_paired(raw_paths, METHODS[0])),
            "primary_method": "routed_selective_ipwm", "diagnostic_method": "selective_ipwm",
            "methods": summaries, "gate_checks": gate_checks, "gate_passed": all(gate_checks.values()),
            "limitations": ["Open-loop prediction only.", "Training seed is the population-level unit."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000); args = parser.parse_args()
    result = summarize(args.inputs, args.draws); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
