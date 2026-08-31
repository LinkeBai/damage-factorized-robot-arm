"""Build a conservative ledger of large reported advantages.

The ledger separates primary-arm, attributable results from useful but generic or
confounded historical results.  It deliberately does not search thresholds,
episodes, or seeds for a favorable subset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def build() -> dict:
    projection = load("results/final/primary-projection-ablation-3seed.json")
    global_ablation = load("results/final/primary-global-matched-ablation-3seed.json")
    confirmation = load("results/final/confirmation-d3-query-seed91031-summary.json")
    synthesis = load("results/final/g2_final_synthesis_5seed.json")

    primary_regret = global_ablation["global_vs_nominal_aggregate"][
        "top1_regret_reduction_percent"]
    confirmation_regret = confirmation["global_vs_nominal_aggregate"][
        "top1_regret_reduction_percent"]
    selective = synthesis["uncertainty"]["coverage_50"]
    ensemble = synthesis["prediction"]["g1_corrected_ensemble_vs_parameter_matched"]

    candidates = [
        {
            "rank": 1,
            "metric": "locked-joint structural violation elimination",
            "advantage_percent": 100.0,
            "scope": "original 5-DoF arm; D2/D4; 3 development seeds",
            "evidence": {
                "without_projection_position_violation_degrees_mean":
                    projection["without_projection_position_violation_degrees"]["mean"],
                "without_projection_velocity_violation_rad_s_mean":
                    projection["without_projection_velocity_violation_rad_s"]["mean"],
                "with_projection_zero_violation_all_seeds":
                    projection["with_projection_zero_violation_all_seeds"],
            },
            "status": "PRIMARY_ATTRIBUTABLE",
            "paper_role": "headline correctness guarantee; not a task-success claim",
        },
        {
            "rank": 2,
            "metric": "top-1 action-regret reduction",
            "advantage_percent": primary_regret["mean"],
            "scope": "original 5-DoF arm; D2/D4; frozen 128-candidate protocol",
            "evidence": {
                "positive_seeds": primary_regret["positive_seeds"],
                "total_seeds": primary_regret["total_seeds"],
                "minimum_seed_improvement_percent": primary_regret["minimum"],
                "maximum_seed_improvement_percent": primary_regret["maximum"],
            },
            "status": "PRIMARY_CONTROL_RELEVANT",
            "paper_role": "headline learned benefit; attribute to matched global residual, not selective IPWM",
        },
        {
            "rank": 3,
            "metric": "selective-prediction RMSE reduction at 50% coverage",
            "advantage_percent": selective["mean_reduction_pct"],
            "scope": "historical five-seed rollout mixture",
            "evidence": {
                "seeds": selective["n_seeds"],
                "std_percent": selective["std_reduction_pct"],
                "all_seed_curves_monotone": synthesis["uncertainty"][
                    "all_seeds_coverage_curve_monotone"],
            },
            "status": "SECONDARY_DEPTH_CONFOUNDED",
            "paper_role": "diagnostic/appendix until fixed-depth confirmation is run",
        },
        {
            "rank": 4,
            "metric": "three-member ensemble prediction improvement",
            "advantage_percent": ensemble["mean"],
            "scope": "historical five-seed prediction benchmark",
            "evidence": {
                "bootstrap_95_ci": ensemble["bootstrap_95_ci"],
                "positive_seeds": ensemble["positive_seeds"],
            },
            "status": "GENERIC_BASELINE_NOT_CORE",
            "paper_role": "supporting baseline only; not an IPWM novelty claim",
        },
        {
            "rank": 5,
            "metric": "fresh D3-query top-1 action-regret reduction",
            "advantage_percent": confirmation_regret["mean"],
            "scope": "post-freeze D3 candidate/query resample; 3 frozen model seeds",
            "evidence": {
                "positive_seeds": confirmation_regret["positive_seeds"],
                "total_seeds": confirmation_regret["total_seeds"],
                "minimum_seed_improvement_percent": confirmation_regret["minimum"],
                "maximum_seed_improvement_percent": confirmation_regret["maximum"],
            },
            "status": "TRANSFER_WEAK_NOT_HEADLINE",
            "paper_role": "report completely; does not establish a large transferable gain",
        },
    ]
    return {
        "protocol": "large_advantage_metric_audit_v1",
        "selection_policy": {
            "all_seeds_reported": True,
            "posthoc_threshold_search": False,
            "posthoc_episode_exclusion": False,
            "model_identity_must_match_claim": True,
        },
        "headline_pair": {
            "structural": "100% elimination of measured locked-joint violations",
            "learned": f"{primary_regret['mean']:.2f}% mean top-1-regret reduction (3/3 seeds)",
            "boundary": "The learned gain is from the matched global residual; selective IPWM attribution is No-Go.",
        },
        "candidates": candidates,
        "next_confirmation": {
            "priority": "fixed-depth uncertainty-risk and real-robot paired failure reduction",
            "promotion_gate": "predeclared comparison, all trials/seeds, paired CI, and no threshold retuning",
            "forbidden": [
                "choose only a favorable seed",
                "change success/contact threshold after seeing results",
                "mix simplified-arm and original-arm evidence",
                "claim oracle or rejected samples as deployed success",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/final/large-advantage-metric-audit.json")
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["headline_pair"], indent=2))


if __name__ == "__main__":
    main()
