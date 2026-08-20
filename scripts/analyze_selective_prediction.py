"""Selective prediction experiment.

Uses ensemble disagreement (epistemic uncertainty) as a rejection threshold:
discard predictions above the threshold quantile and measure RMSE only on
retained samples. Plots RMSE vs. coverage fraction.

Gate hypothesis: RMSE drops monotonically as coverage decreases (higher
threshold = more rejections = only easy samples kept). Spearman correlation
between threshold quantile and retained RMSE should be strongly negative.

Outputs:
  results/analysis/selective_prediction_seed{seed}.json
  reports/selective_prediction_seed{seed}.md

Usage:
  python scripts/analyze_selective_prediction.py [--seed 7] [--domain D3__mixed_composition]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.robotarm.training.topology_ensemble import (
    encode_damage_batch,
    conditioning_damages,
    train_topology_ensemble,
)
from src.robotarm.training.sim_protocol import load_g1_protocol
from src.robotarm.training.target_split import load_target_split
from src.robotarm.envs.mujoco_env import MujocoArmEnv
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_push_ensemble_v1.yaml")
RESULTS_DIR = ROOT / "results" / "analysis"
REPORTS_DIR = ROOT / "reports"
HORIZON = 10

# Coverage fractions to evaluate (fraction of samples retained)
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def collect_per_sample_metrics(ensemble, domain, trajectories, joint_ranges, device, horizon=HORIZON):
    """Returns per-prediction (uncertainty, error) pairs.

    One entry per (trajectory, horizon-window, step) triple.
    Uncertainty = epistemic (member disagreement std).
    Error = ensemble-mean vs ground truth RMSE per sample.
    """
    states = torch.stack([t.states for t in trajectories]).to(device)
    actions = torch.stack([t.actions for t in trajectories]).to(device)
    damages = conditioning_damages([domain.damage] * len(trajectories), "constant")
    contexts = [
        encode_damage_batch(member.encoder, damages, joint_ranges, device)
        for member in ensemble
    ]
    horizon = min(horizon, actions.shape[1])

    all_uncertainty = []
    all_error = []

    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predictions = [states[:, start].clone() for _ in ensemble]
        hidden = [None for _ in ensemble]
        for offset in range(horizon):
            means = []
            for index, member in enumerate(ensemble):
                output, hidden[index] = member.world_model.step(
                    predictions[index], actions[:, start + offset], contexts[index], hidden[index]
                )
                predictions[index] = output["mean"]
                means.append(output["mean"])
            stacked = torch.stack(means)
            mean_pred = stacked.mean(dim=0)
            target = states[:, start + offset + 1]
            error = (mean_pred - target).pow(2).mean(dim=-1).sqrt().detach().cpu().numpy()
            uncertainty = stacked.var(dim=0, unbiased=False).mean(dim=-1).sqrt().detach().cpu().numpy()
            all_uncertainty.extend(uncertainty.tolist())
            all_error.extend(error.tolist())

    return np.array(all_uncertainty), np.array(all_error)


def selective_rmse(uncertainty: np.ndarray, error: np.ndarray, coverage: float) -> dict:
    """Compute RMSE on the `coverage` fraction of samples with lowest uncertainty."""
    n_keep = max(1, int(len(uncertainty) * coverage))
    idx = np.argsort(uncertainty)[:n_keep]
    retained_errors = error[idx]
    rmse = float(np.sqrt(np.mean(retained_errors ** 2)))
    mean_u = float(uncertainty[idx].mean())
    return {
        "coverage": coverage,
        "n_retained": n_keep,
        "rmse": rmse,
        "mean_uncertainty_retained": mean_u,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--domain", type=str, default="D3__mixed_composition")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    epochs = int(config["epochs"])
    steps = int(config["steps"])
    members = int(config["members"])
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    print(f"seed={args.seed}  domain={args.domain}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("[train] collecting trajectories …", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    print("[train] ordinary_deep_ensemble …", flush=True)
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=members, epochs=epochs,
        device=device, seed=args.seed, condition_mode="constant",
    )

    domain = next((d for d in protocol.test if d.domain_id == args.domain), None)
    if domain is None:
        raise ValueError(f"domain {args.domain!r} not in test set")

    print(f"[eval] collecting test trajectories for {domain.domain_id} …", flush=True)
    test_trajs = collect_push_domains(
        (domain,),
        trajectories_per_domain=int(config["trajectories_per_test_domain"]),
        steps=steps,
        seed=args.seed * 100_000 + 500,
        targets=evaluation,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    print("[eval] computing per-sample uncertainty/error …", flush=True)
    uncertainty, error = collect_per_sample_metrics(ensemble, domain, test_trajs, ranges, device)

    print(f"\nTotal samples: {len(uncertainty)}")
    print(f"Uncertainty: mean={uncertainty.mean():.4f}  std={uncertainty.std():.4f}")
    print(f"Error:       mean={error.mean():.4f}  std={error.std():.4f}")

    # Spearman globally
    from scipy.stats import spearmanr
    r_global, _ = spearmanr(uncertainty, error)
    print(f"Global uncertainty-error Spearman: {r_global:+.4f}")

    # selective prediction curve
    results = [selective_rmse(uncertainty, error, c) for c in COVERAGES]

    # baseline: full-coverage RMSE
    baseline_rmse = results[0]["rmse"]
    for r in results:
        r["rmse_reduction_pct"] = 100.0 * (baseline_rmse - r["rmse"]) / baseline_rmse

    print("\n=== Selective Prediction Curve ===")
    print(f"{'Coverage':>9}  {'N_kept':>7}  {'RMSE':>8}  {'Reduction':>10}")
    for r in results:
        print(f"{r['coverage']:>9.0%}  {r['n_retained']:>7}  {r['rmse']:>8.4f}  {r['rmse_reduction_pct']:>+9.2f}%")

    # check monotonicity
    rmse_vals = [r["rmse"] for r in results]
    coverage_vals = [r["coverage"] for r in results]
    r_mono, _ = spearmanr(coverage_vals, rmse_vals)
    is_monotone = r_mono > 0.9
    print(f"\nCoverage-RMSE Spearman: {r_mono:+.4f}  ({'monotone OK' if is_monotone else 'NOT monotone FAIL'})")

    # save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "seed": args.seed,
        "domain": args.domain,
        "n_samples": len(uncertainty),
        "baseline_rmse": baseline_rmse,
        "global_uncertainty_error_spearman": float(r_global),
        "coverage_rmse_spearman": float(r_mono),
        "is_monotone": bool(is_monotone),
        "selective_prediction_curve": results,
    }
    json_path = RESULTS_DIR / f"selective_prediction_seed{args.seed}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")

    # markdown report
    lines = [
        "# Selective Prediction Experiment",
        "",
        f"**Seed**: {args.seed}  **Domain**: {args.domain}",
        "",
        "## Setup",
        "",
        "Ensemble disagreement (epistemic uncertainty = member std) is used as a",
        "rejection score. For each coverage fraction, the lowest-uncertainty samples",
        "are retained and RMSE is computed on that subset.",
        "",
        "## Key Metrics",
        "",
        f"- Global uncertainty-error Spearman: **{r_global:+.4f}**",
        f"- Coverage-RMSE Spearman (monotonicity check): **{r_mono:+.4f}**",
        f"- Is monotone (> 0.9): **{'Yes' if is_monotone else 'No'}**",
        f"- Baseline RMSE (100% coverage): **{baseline_rmse:.4f}**",
        "",
        "## Selective Prediction Curve",
        "",
        "| Coverage | N Retained | RMSE | RMSE Reduction |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['coverage']:.0%} | {r['n_retained']} | {r['rmse']:.4f} | {r['rmse_reduction_pct']:+.2f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "If RMSE drops monotonically as coverage decreases, the ensemble uncertainty",
        "successfully identifies hard predictions. At 50% coverage the RMSE reduction",
        "quantifies the practical gain from selective prediction in a deployment context.",
        "",
        "A non-monotone curve suggests uncertainty is not well-calibrated at the",
        "sample level, even if global Spearman is high (depth-index confound).",
    ]
    report_path = REPORTS_DIR / f"selective_prediction_seed{args.seed}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
