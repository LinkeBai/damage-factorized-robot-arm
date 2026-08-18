"""Merge per-seed checkpoints into a full few_shot_results.csv and re-analyze.

The benchmark checkpoints are cumulative (each seed's checkpoint contains all
prior seeds). This merges the 4-seed checkpoint with the seed-51 run and
re-runs the significance analysis on the full 5-seed set.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BASELINES = [
    "topology_only",
    "history_encoder",
    "parameter_matched",
    "monolithic_matched",
    "residual_only",
]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rmse_at_k(rows, model, shot, seed):
    return {
        r["domain"]: float(r["eval_rmse"])
        for r in rows
        if r["model"] == model and int(r["shots"]) == shot and int(r["seed"]) == seed
    }


def paired_bootstrap(diffs, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n_seed, n_domain = diffs.shape
    means = np.empty(n_boot)
    for b in range(n_boot):
        seed_idx = rng.integers(0, n_seed, size=n_seed)
        total = 0.0
        count = 0
        for si in seed_idx:
            dom_idx = rng.integers(0, n_domain, size=n_domain)
            total += diffs[si, dom_idx].sum()
            count += n_domain
        means[b] = total / count
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--four-seed-checkpoint", type=Path, required=True)
    ap.add_argument("--seed51-checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/final/heldout_5seeds_merged.csv"))
    ap.add_argument("--shot", type=int, default=5)
    args = ap.parse_args()

    rows4 = load_rows(args.four_seed_checkpoint)
    rows51 = load_rows(args.seed51_checkpoint)

    # Deduplicate on (seed, domain, model, shots); keep the merged union.
    key = lambda r: (int(r["seed"]), r["domain"], r["model"], int(r["shots"]))
    merged = {}
    for r in rows4 + rows51:
        merged[key(r)] = r
    rows = list(merged.values())

    seeds = sorted({int(r["seed"]) for r in rows})
    domains = sorted({r["domain"] for r in rows})
    print(f"merged: {len(rows)} rows, seeds={seeds}, domains={len(domains)}")

    # Save merged CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out.resolve()}")

    # --- Per-seed RMSE ---
    print(f"\n=== Per-seed RMSE (K={args.shot}) ===")
    for seed in seeds:
        parts = []
        for m in ["dfwm"] + BASELINES:
            vals = list(rmse_at_k(rows, m, args.shot, seed).values())
            if vals:
                parts.append(f"{m.split('_')[0]}={np.mean(vals):.4f}")
        print(f"seed {seed}: " + "  ".join(parts))

    # --- Significance ---
    print(f"\n=== DFWM vs baseline (K={args.shot}) ===")
    n_domain = len(domains)
    for base in BASELINES:
        diffs = np.zeros((len(seeds), n_domain))
        wins = []
        for si, seed in enumerate(seeds):
            df = rmse_at_k(rows, "dfwm", args.shot, seed)
            bf = rmse_at_k(rows, base, args.shot, seed)
            sw = 0
            for di, dom in enumerate(domains):
                if dom in df and dom in bf:
                    diffs[si, di] = bf[dom] - df[dom]
                    if bf[dom] > df[dom]:
                        sw += 1
            wins.append(sw)
        mean_diff = float(diffs.mean())
        lo, hi = paired_bootstrap(diffs)
        n_better = sum(1 for d in diffs.mean(axis=1) if d > 0)
        sig = "SIGNIFICANT" if lo > 0 else ("MARGINAL" if mean_diff > 0 else "not sig")
        print(f"{base:20s} mean_diff={mean_diff:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"dfwm_better={n_better}/{len(seeds)}  {sig}  wins={wins}")


if __name__ == "__main__":
    main()
