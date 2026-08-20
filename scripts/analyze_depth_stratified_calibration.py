"""Depth-stratified calibration analysis.

Investigates why depth_stratified_spearman (~0.3) diverges from
uncertainty_error_spearman (~0.9).

Hypothesis: uncertainty grows with depth (deeper = more divergence),
error also grows with depth, but within each depth slice the two are
nearly uncorrelated because both are driven by the same depth index.
The per-depth correlation is low because variance within a single step
is small relative to cross-depth variance.

Outputs:
  results/analysis/depth_stratified_calibration.json
  reports/depth_stratified_calibration.md

Usage:
  python scripts/analyze_depth_stratified_calibration.py [--seed 7]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.robotarm.training.topology_ensemble import (
    evaluate_topology_ensemble,
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 4:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def analyze_depth_calibration(
    ensemble, domain, trajectories, joint_ranges, device, horizon=HORIZON
):
    """Re-run evaluation collecting full per-sample (uncertainty, error, depth) triples."""
    from src.robotarm.training.topology_ensemble import encode_damage_batch, conditioning_damages

    states = torch.stack([t.states for t in trajectories]).to(device)
    actions = torch.stack([t.actions for t in trajectories]).to(device)
    damages = conditioning_damages([domain.damage] * len(trajectories), "constant")
    contexts = [
        encode_damage_batch(member.encoder, damages, joint_ranges, device)
        for member in ensemble
    ]

    horizon = min(horizon, actions.shape[1])
    records = []  # (depth, uncertainty, error) per sample per step

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
            for i in range(len(error)):
                records.append({
                    "depth": offset,
                    "uncertainty": float(uncertainty[i]),
                    "error": float(error[i]),
                })

    return records


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
    print(f"seed={args.seed}  device={device}", flush=True)

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

    # find the requested domain
    domain = next((d for d in protocol.test if d.domain_id == args.domain), None)
    if domain is None:
        raise ValueError(f"domain {args.domain!r} not in test set: {[d.domain_id for d in protocol.test]}")

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

    print("[eval] collecting per-sample depth records …", flush=True)
    records = analyze_depth_calibration(ensemble, domain, test_trajs, ranges, device)

    # aggregate
    depths = sorted(set(r["depth"] for r in records))
    depth_stats = {}
    for d in depths:
        subset = [r for r in records if r["depth"] == d]
        u = np.array([r["uncertainty"] for r in subset])
        e = np.array([r["error"] for r in subset])
        depth_stats[d] = {
            "n": len(subset),
            "mean_uncertainty": float(u.mean()),
            "mean_error": float(e.mean()),
            "std_uncertainty": float(u.std()),
            "std_error": float(e.std()),
            "spearman": _spearman(u, e),
        }

    # global (all depths merged)
    all_u = np.array([r["uncertainty"] for r in records])
    all_e = np.array([r["error"] for r in records])
    global_spearman = _spearman(all_u, all_e)
    depth_stratified_mean = float(np.mean([depth_stats[d]["spearman"] for d in depths]))

    print("\n=== Depth-Stratified Calibration ===")
    print(f"Global (merged) Spearman:       {global_spearman:+.4f}")
    print(f"Depth-stratified mean Spearman: {depth_stratified_mean:+.4f}")
    print()
    print(f"{'Depth':>5}  {'N':>5}  {'mean_U':>8}  {'mean_E':>8}  {'Spearman':>9}")
    for d in depths:
        s = depth_stats[d]
        print(f"{d:>5}  {s['n']:>5}  {s['mean_uncertainty']:>8.4f}  {s['mean_error']:>8.4f}  {s['spearman']:>+9.4f}")

    # save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "seed": args.seed,
        "domain": args.domain,
        "global_spearman": global_spearman,
        "depth_stratified_mean_spearman": depth_stratified_mean,
        "depth_stats": {str(d): depth_stats[d] for d in depths},
        "n_records": len(records),
    }
    json_path = RESULTS_DIR / f"depth_stratified_calibration_seed{args.seed}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")

    # markdown report
    lines = [
        "# Depth-Stratified Calibration Analysis",
        "",
        f"**Seed**: {args.seed}  **Domain**: {args.domain}",
        "",
        "## Key Finding",
        "",
        f"- Global Spearman (all depths merged): **{global_spearman:+.4f}**",
        f"- Depth-stratified mean Spearman: **{depth_stratified_mean:+.4f}**",
        "",
        "## Interpretation",
        "",
        "The depth-stratified Spearman is computed per depth-step, then averaged.",
        "Within each step, both uncertainty and error are concentrated in a narrow",
        "range — the cross-depth variance (which drives the global correlation) is",
        "removed. The remaining within-step variance may be much lower, producing a",
        "low per-depth correlation even when the overall calibration is strong.",
        "",
        "## Per-Depth Statistics",
        "",
        "| Depth | N | Mean U | Mean E | Spearman |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in depths:
        s = depth_stats[d]
        lines.append(
            f"| {d} | {s['n']} | {s['mean_uncertainty']:.4f} | {s['mean_error']:.4f} | {s['spearman']:+.4f} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "If within-step Spearman is low but mean_U and mean_E both grow with depth,",
        "this confirms the global correlation is depth-index driven (a confound),",
        "not evidence of calibrated uncertainty. The ensemble disagreement tracks",
        "rollout horizon but not individual prediction difficulty within a step.",
    ]
    report_path = REPORTS_DIR / f"depth_stratified_calibration_seed{args.seed}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
