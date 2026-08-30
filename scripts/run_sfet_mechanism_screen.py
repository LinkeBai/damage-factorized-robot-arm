"""Deterministic few-shot mechanism screen for SFET.

This is deliberately a mechanism diagnostic, not robot-task evidence.  It asks
whether a nominal response prior plus masked Broyden transport is more sample
efficient than fitting a response matrix from the same fault observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robotarm.models.structured_fault_effect_transport import (
    SFETConfig,
    StructuredFaultEffectTransport,
)


def ridge_fit(actions: np.ndarray, effects: np.ndarray, ridge: float) -> np.ndarray:
    gram = actions.T @ actions + ridge * np.eye(actions.shape[1])
    return effects.T @ actions @ np.linalg.inv(gram)


def trial(seed: int, shots: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    effect_dim, action_dim, locked = 2, 5, 2
    nominal = rng.normal(0.0, 0.55, size=(effect_dim, action_dim))
    true = nominal + rng.normal(0.0, 0.22, size=nominal.shape)
    true[:, locked] = 0.0
    mask = np.ones(action_dim)
    mask[locked] = 0.0
    train_a = rng.normal(size=(shots, action_dim)) * mask
    train_e = train_a @ true.T
    test_a = rng.normal(size=(256, action_dim)) * mask
    test_e = test_a @ true.T

    masked_nominal = nominal.copy()
    masked_nominal[:, locked] = 0.0
    empirical = ridge_fit(train_a, train_e, ridge=1e-3)
    empirical[:, locked] = 0.0
    sfet = StructuredFaultEffectTransport(
        nominal, locked=(locked,), config=SFETConfig(secant_epsilon=1e-12)
    )
    for action, effect in zip(train_a, train_e):
        sfet.update(action, effect)

    def rmse(matrix: np.ndarray) -> float:
        return float(np.sqrt(np.mean((test_a @ matrix.T - test_e) ** 2)))

    base = rmse(masked_nominal)
    sfet_error = rmse(sfet.jacobian)
    return {
        "masked_nominal_rmse": base,
        "empirical_ridge_rmse": rmse(empirical),
        "sfet_rmse": sfet_error,
        "sfet_relative_reduction_vs_masked_nominal": (base - sfet_error) / base,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        {"seed": seed, "shots": shots, **trial(seed, shots)}
        for shots in (1, 3, 5)
        for seed in (7, 17, 27)
    ]
    summary = {}
    for shots in (1, 3, 5):
        selected = [row for row in rows if row["shots"] == shots]
        summary[str(shots)] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "masked_nominal_rmse",
                "empirical_ridge_rmse",
                "sfet_rmse",
                "sfet_relative_reduction_vs_masked_nominal",
            )
        }
    result = {
        "diagnostic": "sfet_linear_response_mechanism_screen",
        "evidence_scope": "synthetic linear response; not robot-task evidence",
        "rows": rows,
        "mean_by_shots": summary,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
