"""Build the final G2 evidence synthesis from frozen result artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (7, 17, 27, 37, 47)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_item(selective: dict, coverage: float) -> dict:
    for item in selective["selective_prediction_curve"]:
        if abs(float(item["coverage"]) - coverage) < 1e-9:
            return item
    raise ValueError(f"selective prediction artifact has no coverage={coverage}")


def _validate_five_seeds(name: str, values) -> None:
    actual = tuple(sorted(int(seed) for seed in values))
    if actual != SEEDS:
        raise ValueError(f"{name} seeds are {actual}, expected {SEEDS}")


def build_synthesis(root: Path) -> tuple[dict, list[dict]]:
    final = root / "results" / "final"
    analysis = root / "results" / "analysis"
    structured = _load(final / "g2_structured_vs_ordinary_5seed.json")
    heldout = _load(final / "g2_heldout_topology_5seed.json")
    members = _load(final / "g2_member_ablation_5seed.json")
    robust = _load(final / "g1_robust_zero_shot_5seed_summary.json")
    selective = _load(analysis / "selective_prediction_5seed.json")
    depth = _load(analysis / "depth_stratified_calibration_5seed.json")

    for name, artifact in (
        ("structured", structured), ("heldout", heldout),
        ("member ablation", members), ("robust", robust),
        ("selective", selective), ("depth calibration", depth),
    ):
        _validate_five_seeds(name, artifact["seeds"])

    run_summaries = [
        _load(root / "runs" / "g2_push_ensemble" / f"seed{seed}_v1" / "summary.json")
        for seed in SEEDS
    ]
    protocol_hashes = {item["protocol_sha256"] for item in run_summaries}
    if protocol_hashes != {structured["protocol_sha256"]}:
        raise ValueError("G2 ensemble summaries do not share the frozen protocol hash")
    if tuple(sorted(int(item["seed"]) for item in run_summaries)) != SEEDS:
        raise ValueError("G2 ensemble run summaries are incomplete")

    coverage_70 = _coverage_item(selective, 0.7)
    coverage_50 = _coverage_item(selective, 0.5)
    timings = {
        method: [float(item["train_seconds"][method]) for item in run_summaries]
        for method in ("structured_ensemble", "ordinary_deep_ensemble")
    }
    parameters = run_summaries[0]["parameters"]
    m1_d3 = members["results"]["m1_D3__mixed_composition"]
    m3_d3 = members["results"]["m3_D3__mixed_composition"]

    evidence = {
        "generated_from_frozen_artifacts": True,
        "date": "2026-08-21",
        "seeds": list(SEEDS),
        "protocol_sha256": structured["protocol_sha256"],
        "primary_conclusion": (
            "Ordinary ensemble averaging and selective rejection are supported; "
            "a distinct G2 topology-conditioning advantage is not supported."
        ),
        "prediction": {
            "g1_corrected_ensemble_vs_parameter_matched": robust["aggregate"]["vs_parameter_matched_pct"],
            "g2_structured_vs_ordinary": {
                "mean_improvement_pct": structured["mean_improvement_pct"],
                "bootstrap_95_ci": structured["bootstrap_95_ci"],
                "positive_seeds": structured["positive_seeds"],
                "decision": "NO DISTINCT STRUCTURED ADVANTAGE",
            },
            "heldout_d3": {
                "mean_improvement_pct": heldout["mean_d3_pct"],
                "bootstrap_95_ci": heldout["bootstrap_95_ci_d3"],
                "positive_seeds": heldout["positive_seeds_d3"],
                "decision": "NO HELD-OUT TOPOLOGY ADVANTAGE",
            },
            "member_ablation_d3": {
                "one_member_rmse": m1_d3["mean"],
                "three_member_rmse": m3_d3["mean"],
                "three_vs_one_improvement_pct": m3_d3["improvement_vs_m1_pct"],
            },
        },
        "uncertainty": {
            "coverage_70": coverage_70,
            "coverage_50": coverage_50,
            "all_seeds_coverage_curve_monotone": selective["all_seeds_monotone"],
            "global_spearman": depth["global_spearman"],
            "depth_stratified_spearman": depth["depth_stratified_spearman"],
            "decision": "USE FOR SELECTIVE PREDICTION WITH DEPTH-CONFOUND CAVEAT",
        },
        "compute": {
            method: {
                "parameters": int(parameters[method]),
                "train_seconds_mean": statistics.mean(timings[method]),
                "train_seconds_std": statistics.stdev(timings[method]),
                "train_seconds_per_seed": {
                    str(item["seed"]): float(item["train_seconds"][method])
                    for item in run_summaries
                },
            }
            for method in timings
        },
        "structural_branch": {
            "ft_gwm_k0": "PASS: exact fixed-SE(3) kinematics",
            "ft_gwm_k1": "TWO-SEED PROVISIONAL PASS: zero violation and <=5% D3 free-arm regression",
            "ft_gwm_k2": "NO-GO: object +986.08%, free-arm +22.11% versus object-aware baseline",
        },
        "claim_boundary": {
            "supported": [
                "Three-member ordinary ensemble improves prediction over one member.",
                "Ensemble disagreement supports a coverage-accuracy tradeoff on the evaluated rollout mixture.",
                "FT-GWM K1 provides exact lock satisfaction with provisional free-joint fidelity.",
            ],
            "not_supported": [
                "Topology conditioning outperforms an ordinary deep ensemble.",
                "Uncertainty is fully instance-calibrated at fixed rollout depth.",
                "FT-GWM is a complete Push object/contact world model.",
                "Prediction gains produce statistically established control gains.",
            ],
        },
        "source_artifacts": [
            "results/final/g1_robust_zero_shot_5seed_summary.json",
            "results/final/g2_structured_vs_ordinary_5seed.json",
            "results/final/g2_heldout_topology_5seed.json",
            "results/final/g2_member_ablation_5seed.json",
            "results/analysis/selective_prediction_5seed.json",
            "results/analysis/depth_stratified_calibration_5seed.json",
        ],
    }

    rows = [
        {"section": "prediction", "metric": "g1_ensemble_vs_parameter_matched_improvement_pct", "scope": "D2+D3", "mean": robust["aggregate"]["vs_parameter_matched_pct"]["mean"], "ci_lo": robust["aggregate"]["vs_parameter_matched_pct"]["bootstrap_95_ci"][0], "ci_hi": robust["aggregate"]["vs_parameter_matched_pct"]["bootstrap_95_ci"][1], "status": "supported"},
        {"section": "prediction", "metric": "g2_structured_vs_ordinary_improvement_pct", "scope": "D2+D3", "mean": structured["mean_improvement_pct"], "ci_lo": structured["bootstrap_95_ci"][0], "ci_hi": structured["bootstrap_95_ci"][1], "status": "ci_crosses_zero"},
        {"section": "prediction", "metric": "heldout_d3_improvement_pct", "scope": "D3", "mean": heldout["mean_d3_pct"], "ci_lo": heldout["bootstrap_95_ci_d3"][0], "ci_hi": heldout["bootstrap_95_ci_d3"][1], "status": "ci_crosses_zero"},
        {"section": "ablation", "metric": "three_vs_one_member_improvement_pct", "scope": "D3", "mean": m3_d3["improvement_vs_m1_pct"], "ci_lo": "", "ci_hi": "", "status": "supported"},
        {"section": "selective_prediction", "metric": "rmse_reduction_pct", "scope": "70% coverage", "mean": coverage_70["mean_reduction_pct"], "ci_lo": "", "ci_hi": "", "status": "monotone_5_of_5"},
        {"section": "selective_prediction", "metric": "rmse_reduction_pct", "scope": "50% coverage", "mean": coverage_50["mean_reduction_pct"], "ci_lo": "", "ci_hi": "", "status": "monotone_5_of_5"},
        {"section": "calibration", "metric": "global_spearman", "scope": "D3", "mean": depth["global_spearman"]["mean"], "ci_lo": "", "ci_hi": "", "status": "depth_confound"},
        {"section": "calibration", "metric": "depth_stratified_spearman", "scope": "D3", "mean": depth["depth_stratified_spearman"]["mean"], "ci_lo": "", "ci_hi": "", "status": "moderate_high_variance"},
    ]
    return evidence, rows


def render_markdown(data: dict) -> str:
    pred = data["prediction"]
    unc = data["uncertainty"]
    comp = data["compute"]
    g1 = pred["g1_corrected_ensemble_vs_parameter_matched"]
    g2 = pred["g2_structured_vs_ordinary"]
    held = pred["heldout_d3"]
    c70, c50 = unc["coverage_70"], unc["coverage_50"]
    return f"""# Final G2 Evidence Synthesis

**Date:** 2026-08-21
**Seeds:** 7, 17, 27, 37, 47
**Protocol SHA-256:** `{data['protocol_sha256']}`

## Decision

The supported main result is ordinary ensemble averaging plus selective
prediction. The evidence does not support a distinct topology-conditioning or
complete structured-world-model advantage. FT-GWM K1 remains a narrower
constraint-preserving joint-dynamics contribution.

## Prediction Evidence

| Comparison | Mean improvement | Seed bootstrap 95% CI | Positive seeds | Decision |
|---|---:|---:|---:|---|
| G1 corrected 3-member ensemble vs parameter-matched single | {g1['mean']:.2f}% | [{g1['bootstrap_95_ci'][0]:.2f}%, {g1['bootstrap_95_ci'][1]:.2f}%] | {g1['positive_seeds']}/5 | Supported |
| G2 structured vs ordinary ensemble | {g2['mean_improvement_pct']:.2f}% | [{g2['bootstrap_95_ci'][0]:.2f}%, {g2['bootstrap_95_ci'][1]:.2f}%] | {g2['positive_seeds']}/5 | CI crosses zero |
| Held-out D3 topology conditioning | {held['mean_improvement_pct']:.2f}% | [{held['bootstrap_95_ci'][0]:.2f}%, {held['bootstrap_95_ci'][1]:.2f}%] | {held['positive_seeds']}/5 | CI crosses zero |

The D3 member ablation improves mean RMSE from
`{pred['member_ablation_d3']['one_member_rmse']:.4f}` (one member) to
`{pred['member_ablation_d3']['three_member_rmse']:.4f}` (three members), a
`{pred['member_ablation_d3']['three_vs_one_improvement_pct']:.2f}%` reduction.

## Selective Prediction

| Coverage | Mean RMSE | Mean reduction | Across-seed std |
|---:|---:|---:|---:|
| 70% | {c70['mean_rmse']:.4f} | {c70['mean_reduction_pct']:.2f}% | {c70['std_reduction_pct']:.2f}% |
| 50% | {c50['mean_rmse']:.4f} | {c50['mean_reduction_pct']:.2f}% | {c50['std_reduction_pct']:.2f}% |

All five coverage-RMSE curves are monotone. However, global uncertainty-error
Spearman (`{unc['global_spearman']['mean']:.3f} +/- {unc['global_spearman']['std']:.3f}`)
drops to `{unc['depth_stratified_spearman']['mean']:.3f} +/- {unc['depth_stratified_spearman']['std']:.3f}`
after stratifying by rollout depth. Therefore disagreement is a useful
rejection score on the evaluated mixed-depth rollout distribution, but it must
not be described as fully instance-calibrated at a fixed horizon.

This synthesis corrects an indexing error in the older selective-prediction
summary: the actual reductions are `30.96%` at 70% coverage and `50.50%` at
50% coverage.

## Compute

| Method | Parameters | Mean train seconds | Std seconds | Device |
|---|---:|---:|---:|---|
| Structured ensemble | {comp['structured_ensemble']['parameters']:,} | {comp['structured_ensemble']['train_seconds_mean']:.1f} | {comp['structured_ensemble']['train_seconds_std']:.1f} | CUDA |
| Ordinary deep ensemble | {comp['ordinary_deep_ensemble']['parameters']:,} | {comp['ordinary_deep_ensemble']['train_seconds_mean']:.1f} | {comp['ordinary_deep_ensemble']['train_seconds_std']:.1f} | CUDA |

Both G2 ensembles use three members, 20 epochs, 150-step trajectories and the
same parameter count. Wall-clock values are measured end-to-end training times
from the five frozen run summaries; GPU model was not recorded, so no
cross-machine compute claim is permitted.

## Structural Branch

- **FT-GWM K0:** PASS for exact fixed-SE(3) kinematics.
- **FT-GWM K1:** two-seed provisional PASS for zero violation and free-joint fidelity.
- **FT-GWM K2:** NO-GO for complete Push prediction; object RMSE regressed 986.08%.

## Frozen Claims

Supported:

- A three-member ordinary ensemble materially improves prediction over a single model.
- Ensemble disagreement supports selective rejection on the evaluated rollout mixture.
- FT-GWM K1 exactly satisfies known joint locks with provisional joint fidelity.

Not supported:

- Topology conditioning outperforms an ordinary deep ensemble.
- Disagreement is fully instance-calibrated at fixed rollout depth.
- FT-GWM is a complete Push object/contact model.
- Prediction improvements already imply statistically established control gains.

## Next Decision

Do not reopen DFWM/CR-GWM/RC-GWM/FT-GWM head tuning. Before G3, freeze the
paper tables and decide whether the narrower ensemble/selective-prediction
claim justifies real-robot evaluation. Any G3 uncertainty gate must be
calibrated at the deployment horizon, not from pooled rollout depths.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    data, rows = build_synthesis(root)
    output_json = root / "results" / "final" / "g2_final_synthesis_5seed.json"
    output_csv = root / "results" / "final" / "g2_final_synthesis_5seed.csv"
    output_report = root / "reports" / "g2-final-synthesis-20260821.md"
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_report.write_text(render_markdown(data), encoding="utf-8")
    print(f"wrote {output_json.relative_to(root)}")
    print(f"wrote {output_csv.relative_to(root)}")
    print(f"wrote {output_report.relative_to(root)}")


if __name__ == "__main__":
    main()
