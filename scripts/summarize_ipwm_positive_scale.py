"""Audit and summarize the frozen historical-positive replication after all jobs finish."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ipwm_positive_scale_20260911"
OLD = ROOT / "runs/ipwm_targeted_advantage_20260911/legacy_policy_100"
PHYSICS = ["nominal", "weak_motor", "high_damping", "delay_1", "noisy_deadband",
           "mixed_composition", "mixed_unseen"]
PRIMARY = ["high_damping", "mixed_composition", "mixed_unseen"]
SEEDS = [27, 37, 47]
METHODS = ["carrier", "full", "selective"]
HORIZONS = [10, 25, 50]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def old_fingerprints() -> set[str]:
    import torch
    values = set()
    for path in OLD.glob("*.pt"):
        for trajectory in torch.load(path, weights_only=False):
            state = trajectory.states[0].cpu().numpy().astype(np.float32)
            action = trajectory.actions.cpu().numpy().astype(np.float32)
            values.add(hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest())
    return values


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["independent_trajectories_total"] == 8400
    fingerprints, data_hashes = [], {}
    for physics in PHYSICS:
        for shard in range(12):
            data = OUT / "data" / physics / f"shard-{shard:02d}.pt"
            record_path = data.with_suffix(".json")
            record = json.loads(record_path.read_text())
            assert record["physics"] == physics and record["shard"] == shard
            assert record["trajectories"] == 100 and sha(data) == record["sha256"]
            fingerprints.extend(record["fingerprints"]); data_hashes[str(data.relative_to(ROOT))] = record["sha256"]
    assert len(fingerprints) == 8400 and len(set(fingerprints)) == 8400
    overlap = set(fingerprints) & old_fingerprints()
    assert not overlap

    grouped = defaultdict(list)
    source_hashes = {}
    for seed in SEEDS:
        for physics in PHYSICS:
            for shard in range(12):
                path = OUT / "evaluation" / f"seed{seed}" / physics / f"shard-{shard:02d}.json"
                doc = json.loads(path.read_text())
                assert (doc["seed"], doc["physics"], doc["shard"]) == (seed, physics, shard)
                assert doc["data_sha256"] == data_hashes[str((OUT / "data" / physics / f"shard-{shard:02d}.pt").relative_to(ROOT))]
                source_hashes[str(path.relative_to(ROOT))] = sha(path)
                rows = doc["rows"]
                for row in rows:
                    if row["method"] in METHODS:
                        grouped[(seed, physics, row["horizon"], row["method"])].append(row)

    expected_windows = {10: 18000, 25: 7200, 50: 3600}
    table = []
    for seed in SEEDS:
        for physics in PHYSICS:
            for horizon in HORIZONS:
                metrics = {}
                for method in METHODS:
                    rows = grouped[(seed, physics, horizon, method)]
                    assert len(rows) == expected_windows[horizon]
                    for key in ["object_mse", "object_xy_mse", "object_velocity_mse", "free_mse",
                                "robot_carrier_max", "lock_violation"]:
                        values = np.asarray([r[key] for r in rows], dtype=np.float64)
                        assert np.isfinite(values).all()
                        metrics[(method, key)] = values
                assert np.allclose(metrics[("full", "object_mse")], metrics[("selective", "object_mse")], atol=1e-12)
                assert metrics[("selective", "robot_carrier_max")].max() <= 1e-8
                assert metrics[("selective", "lock_violation")].max() <= 1e-8
                carrier = np.sqrt(metrics[("carrier", "object_mse")].mean())
                selective = np.sqrt(metrics[("selective", "object_mse")].mean())
                table.append({
                    "seed": seed, "physics": physics, "horizon": horizon,
                    "independent_trajectories": 1200, "windows": expected_windows[horizon],
                    "carrier_object_state_rmse": carrier,
                    "selective_object_state_rmse": selective,
                    "object_state_improvement_pct": 100 * (carrier - selective) / carrier,
                    "carrier_xy_rmse_m": np.sqrt(metrics[("carrier", "object_xy_mse")].mean()),
                    "selective_xy_rmse_m": np.sqrt(metrics[("selective", "object_xy_mse")].mean()),
                    "carrier_velocity_rmse_mps": np.sqrt(metrics[("carrier", "object_velocity_mse")].mean()),
                    "selective_velocity_rmse_mps": np.sqrt(metrics[("selective", "object_velocity_mse")].mean()),
                    "selective_free_robot_rmse": np.sqrt(metrics[("selective", "free_mse")].mean()),
                    "max_robot_change_from_carrier": metrics[("selective", "robot_carrier_max")].max(),
                    "max_lock_violation": metrics[("selective", "lock_violation")].max(),
                })
    assert len(table) == 63
    write_csv(OUT / "all-cells.csv", table)
    primary = [r for r in table if r["physics"] in PRIMARY]
    h50 = [r for r in primary if r["horizon"] == 50]
    summary = {
        "independent_trajectories": 8400,
        "independent_trajectories_per_condition": 1200,
        "all_cells": len(table), "primary_cells": len(primary),
        "positive_primary_cells": int(sum(r["object_state_improvement_pct"] > 0 for r in primary)),
        "mean_primary_improvement_pct": float(np.mean([r["object_state_improvement_pct"] for r in primary])),
        "positive_primary_h50_cells": int(sum(r["object_state_improvement_pct"] > 0 for r in h50)),
        "mean_primary_h50_improvement_pct": float(np.mean([r["object_state_improvement_pct"] for r in h50])),
        "max_robot_change_from_carrier": float(max(r["max_robot_change_from_carrier"] for r in table)),
        "max_lock_violation": float(max(r["max_lock_violation"] for r in table)),
        "dataset_unique": True, "overlap_with_previous_700": 0,
        "protocol_sha256": sha(OUT / "protocol.json"), "evaluation_hashes": source_hashes,
        "interpretation": "Mixed-unit object-state RMSE is retained for historical comparability; xy and velocity metrics must be reported separately. Model seeds are fixed checkpoints, not independent retraining repetitions."
    }
    (OUT / "results-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation_hashes"}, indent=2))


if __name__ == "__main__":
    main()
