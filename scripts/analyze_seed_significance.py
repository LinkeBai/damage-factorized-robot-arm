"""Per-seed breakdown and significance test for the 6-method benchmark.

Answers two questions for the paper's Go gate:
1. Is DFWM better than each baseline on at least 2/3 seeds? (G1 Go condition)
2. Is the DFWM-vs-baseline gap statistically meaningful? (paired bootstrap 95% CI)
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


def rmse_at_k(rows: list[dict], model: str, shot: int, seed: int, metric: str) -> dict[str, float]:
    """Return {domain: rmse} for a given model/shot/seed."""
    out = {}
    for r in rows:
        if r["model"] == model and int(r["shots"]) == shot and int(r["seed"]) == seed:
            if r.get(metric, "") != "":
                out[r["domain"]] = float(r[metric])
    return out


def paired_bootstrap(diffs: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Hierarchical-ish paired bootstrap: resample domains within each seed.

    ``diffs`` has shape (n_seed, n_domain) = dfwm_baseline - dfwm (positive = dfwm better).
    We resample seeds with replacement (outer), then domains within each selected
    seed (inner), matching the plan's hierarchical paired bootstrap.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path, help="Path to few_shot_results.csv")
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--metric", choices=("eval_rmse", "multi_step_rmse"), default="eval_rmse")
    args = ap.parse_args()

    rows = load_rows(args.csv_path)
    seeds = sorted({int(r["seed"]) for r in rows})
    models = sorted({r["model"] for r in rows})
    domains = sorted({r["domain"] for r in rows})
    print(f"seeds={seeds}, models={len(models)}, domains={len(domains)}, K={args.shot}, metric={args.metric}\n")

    # --- 1. Per-seed breakdown ---
    print("=== Per-seed RMSE (K={}) ===".format(args.shot))
    for seed in seeds:
        line = f"seed {seed}: "
        parts = []
        for m in ["dfwm"] + BASELINES:
            vals = list(rmse_at_k(rows, m, args.shot, seed, args.metric).values())
            if vals:
                parts.append(f"{m.split('_')[0]}={np.mean(vals):.4f}")
        line += "  ".join(parts)
        print(line)

    # --- 2. DFWM-wins count per baseline + bootstrap CI ---
    print(f"\n=== DFWM vs baseline (K={args.shot}), paired across seed x domain ===")
    n_domain = len(domains)
    for base in BASELINES:
        if base not in models:
            continue
        diffs = np.zeros((len(seeds), n_domain))
        wins_by_seed = []
        for si, seed in enumerate(seeds):
            df = rmse_at_k(rows, "dfwm", args.shot, seed, args.metric)
            bf = rmse_at_k(rows, base, args.shot, seed, args.metric)
            seed_wins = 0
            for di, dom in enumerate(domains):
                if dom in df and dom in bf:
                    diffs[si, di] = bf[dom] - df[dom]  # positive = dfwm better
                    if bf[dom] > df[dom]:
                        seed_wins += 1
            wins_by_seed.append(seed_wins)
        mean_diff = float(diffs.mean())
        lo, hi = paired_bootstrap(diffs)
        n_seed_better = sum(1 for d in diffs.mean(axis=1) if d > 0)
        sig = "SIGNIFICANT" if lo > 0 else ("MARGINAL" if mean_diff > 0 else "not significant")
        print(
            f"{base:20s}: mean_diff={mean_diff:+.4f}  "
            f"95% CI=[{lo:+.4f}, {hi:+.4f}]  "
            f"dfwm better in {n_seed_better}/{len(seeds)} seeds  {sig}"
        )
        print(f"{'':20s}  wins-per-seed (out of {n_domain} domains): {wins_by_seed}")


if __name__ == "__main__":
    main()
