"""Aggregate frozen G2 structured-vs-ordinary ensemble runs by seed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    args = parser.parse_args()

    records = []
    protocol_hashes = set()
    for path in sorted(args.runs_dir.glob("seed*_v1/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol_hashes.add(payload["protocol_sha256"])
        if len(set(payload["parameters"].values())) != 1:
            raise ValueError(f"parameter mismatch in {path}")
        by_domain = {}
        for row in payload["rows"]:
            by_domain.setdefault(row["domain"], {})[row["method"]] = row
        for domain, methods in by_domain.items():
            structured = methods["structured_ensemble"]["ensemble_rmse"]
            ordinary = methods["ordinary_deep_ensemble"]["ensemble_rmse"]
            records.append({
                "seed": payload["seed"],
                "domain": domain,
                "structured_rmse": structured,
                "ordinary_rmse": ordinary,
                "improvement_pct": 100.0 * (ordinary - structured) / ordinary,
            })
    if len(protocol_hashes) != 1:
        raise ValueError(f"runs use different protocol hashes: {protocol_hashes}")

    seeds = sorted({row["seed"] for row in records})
    per_seed = np.array([
        np.mean([row["improvement_pct"] for row in records if row["seed"] == seed])
        for seed in seeds
    ])
    rng = np.random.default_rng(20260819)
    samples = per_seed[
        rng.integers(0, len(per_seed), size=(args.bootstrap_samples, len(per_seed)))
    ].mean(axis=1)
    summary = {
        "protocol_sha256": next(iter(protocol_hashes)),
        "seeds": seeds,
        "per_seed_mean_improvement_pct": dict(zip(map(str, seeds), per_seed.tolist())),
        "mean_improvement_pct": float(per_seed.mean()),
        "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "positive_seeds": int((per_seed > 0).sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "seed",
        "rows": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
