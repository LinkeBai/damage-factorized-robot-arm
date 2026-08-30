"""Conformal quantile calibration for the unchanged Z65 context posterior."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.training.sim_protocol import load_g1_protocol
from scripts.calibrate_bt_dpwm_z80_context_posterior import collect_seed
from scripts.run_bt_dpwm_fewshot_z48 import add_compositional_training_domains


def fit_conformal_radii(records, budgets, coverages):
    radii = {}
    for budget in budgets:
        subset = [row for row in records if row["budget"] == budget]
        errors = np.abs(np.asarray([row["normalized_error"] for row in subset]))
        std = np.sqrt(np.maximum(np.asarray(
            [row["normalized_variance"] for row in subset]), 1e-12))
        scores = errors/std
        radii[str(budget)] = {str(float(coverage)): np.quantile(
            scores, float(coverage), axis=0, method="higher").tolist()
            for coverage in coverages}
    return radii


def conformal_coverage(records, radii, budgets, coverages):
    rows = []
    for budget in budgets:
        subset = [row for row in records if row["budget"] == budget]
        errors = np.abs(np.asarray([row["normalized_error"] for row in subset]))
        std = np.sqrt(np.maximum(np.asarray(
            [row["normalized_variance"] for row in subset]), 1e-12))
        scores = errors/std
        for nominal in coverages:
            radius = np.asarray(radii[str(budget)][str(float(nominal))])
            per_dimension = np.mean(scores <= radius, axis=0)
            dimension_errors = np.abs(per_dimension-float(nominal))
            rows.append({"budget": int(budget), "nominal_coverage": float(nominal),
                "empirical_coverage": float(per_dimension.mean()),
                "per_dimension_coverage": per_dimension.tolist(),
                "mean_dimensionwise_absolute_error": float(dimension_errors.mean()),
                "maximum_dimensionwise_absolute_error": float(dimension_errors.max())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z81_conformal_context_calibration_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z81_conformal_context_calibration/summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    encoder_cfg = yaml.safe_load(Path(cfg["encoder_config"]).read_text(encoding="utf-8"))
    parent = yaml.safe_load(Path(encoder_cfg["parent_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(parent["base_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(base["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(q0a["protocol"])
    domains = add_compositional_training_domains(protocol, parent)
    cfg["encoder_hidden_dim"] = int(encoder_cfg["hidden_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development, confirmation = [], []
    for seed in cfg["development_seeds"]:
        print(f"[Z81] collect development seed={seed}", flush=True)
        development.extend(collect_seed(seed, cfg, domains, q0a, device))
    radii = fit_conformal_radii(
        development, cfg["transition_budgets"], cfg["nominal_coverages"])
    for seed in cfg["confirmation_seeds"]:
        print(f"[Z81] collect confirmation seed={seed}", flush=True)
        confirmation.extend(collect_seed(seed, cfg, domains, q0a, device))
    coverage = conformal_coverage(confirmation, radii,
        cfg["transition_budgets"], cfg["nominal_coverages"])
    overall = float(np.mean(
        [row["mean_dimensionwise_absolute_error"] for row in coverage]))
    by_nominal = {str(float(nominal)): float(np.mean([
        row["mean_dimensionwise_absolute_error"] for row in coverage
        if row["nominal_coverage"] == nominal])) for nominal in cfg["nominal_coverages"]}
    by_budget = {str(int(budget)): float(np.mean([
        row["mean_dimensionwise_absolute_error"] for row in coverage
        if row["budget"] == budget])) for budget in cfg["transition_budgets"]}
    gate = {
        "overall_dimensionwise_mace": overall,
        "per_nominal_dimensionwise_mace": by_nominal,
        "per_budget_dimensionwise_mace": by_budget,
        "passed": (overall <= float(cfg["maximum_overall_dimensionwise_mace"])
            and max(by_nominal.values()) <= float(
                cfg["maximum_per_nominal_dimensionwise_mace"])
            and max(by_budget.values()) <= float(
                cfg["maximum_per_budget_dimensionwise_mace"]))}
    output = {"version": cfg["version"], "device": str(device),
        "development_seeds": cfg["development_seeds"],
        "confirmation_seeds": cfg["confirmation_seeds"],
        "domain_count": len(domains), "development_record_count": len(development),
        "confirmation_record_count": len(confirmation),
        "conformal_radii_by_budget_coverage_dimension": radii,
        "confirmation_coverage": coverage, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
