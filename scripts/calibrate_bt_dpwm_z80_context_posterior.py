"""Calibrate Z65 physical-context posterior variance on independent probes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from scipy.stats import norm

from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_bt_dpwm_fewshot_z48 import (
    add_compositional_training_domains, physical_context_batch)
from scripts.train_physical_context_encoder_z64 import CONTEXT_SCALE


def fit_temperature(records, budgets, clip):
    result = {}
    for budget in budgets:
        subset = [row for row in records if row["budget"] == budget]
        errors = np.asarray([row["normalized_error"] for row in subset])
        variances = np.asarray([row["normalized_variance"] for row in subset])
        temperature = np.mean(errors**2/np.maximum(variances, 1e-12), axis=0)
        result[str(budget)] = np.clip(temperature, clip[0], clip[1]).tolist()
    return result


def coverage_summary(records, temperatures, nominal_coverages):
    summaries = []
    for budget in sorted({row["budget"] for row in records}):
        subset = [row for row in records if row["budget"] == budget]
        errors = np.asarray([row["normalized_error"] for row in subset])
        variances = np.asarray([row["normalized_variance"] for row in subset])
        temp = np.asarray(temperatures[str(budget)])
        standardized = np.abs(errors)/np.sqrt(np.maximum(variances*temp, 1e-12))
        for nominal in nominal_coverages:
            radius = norm.ppf((1+float(nominal))/2)
            per_dimension = np.mean(standardized <= radius, axis=0)
            summaries.append({"budget": budget, "nominal_coverage": float(nominal),
                "empirical_coverage": float(per_dimension.mean()),
                "per_dimension_coverage": per_dimension.tolist(),
                "absolute_error": float(abs(per_dimension.mean()-nominal))})
    return summaries


def collect_seed(seed, cfg, domains, q0a, device):
    encoder = UncertainPhysicalContextEncoder(
        hidden_dim=int(cfg["encoder_hidden_dim"])).to(device)
    encoder_path = Path(str(cfg["encoder_run_template"]).format(seed=seed))
    encoder.load_state_dict(torch.load(
        encoder_path/"context_encoder.pt", map_location=device))
    encoder.eval()
    count = int(cfg["trajectories_per_domain"])
    trajectories = collect_push_domains_warp(
        domains, trajectories_per_domain=count, steps=max(cfg["transition_budgets"]),
        seed=seed*10000+int(cfg["collection_seed_offset"]),
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        excitation="active")
    targets = physical_context_batch(domains, device, torch.float32)
    masks, _ = _damage_tensors([domain.damage for domain in domains], device)
    scale = CONTEXT_SCALE.to(device)
    records = []
    with torch.no_grad():
        for repeat in range(count):
            selected = [trajectories[index*count+repeat]
                        for index in range(len(domains))]
            for budget in cfg["transition_budgets"]:
                states = torch.stack([item.states[:budget+1] for item in selected]).to(device)
                actions = torch.stack([item.actions[:budget] for item in selected]).to(device)
                mean, log_variance = encoder(
                    states, actions, masks, return_uncertainty=True)
                errors = ((mean-targets)/scale).cpu().numpy()
                variances = torch.exp(log_variance).cpu().numpy()
                for index, domain in enumerate(domains):
                    records.append({"seed": seed, "repeat": repeat,
                        "domain": domain.domain_id, "topology": domain.topology,
                        "budget": int(budget),
                        "normalized_error": errors[index].tolist(),
                        "normalized_variance": variances[index].tolist()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z80_context_posterior_calibration_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z80_context_posterior_calibration/summary.json"))
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
        print(f"[Z80] collect development seed={seed}", flush=True)
        development.extend(collect_seed(seed, cfg, domains, q0a, device))
    temperatures = fit_temperature(
        development, cfg["transition_budgets"], cfg["temperature_clip"])
    for seed in cfg["confirmation_seeds"]:
        print(f"[Z80] collect confirmation seed={seed}", flush=True)
        confirmation.extend(collect_seed(seed, cfg, domains, q0a, device))
    dev_coverage = coverage_summary(
        development, temperatures, cfg["nominal_coverages"])
    confirmation_coverage = coverage_summary(
        confirmation, temperatures, cfg["nominal_coverages"])
    confirmation_mace = float(np.mean(
        [row["absolute_error"] for row in confirmation_coverage]))
    output = {"version": cfg["version"], "device": str(device),
              "context_scale": CONTEXT_SCALE.tolist(),
              "development_seeds": cfg["development_seeds"],
              "confirmation_seeds": cfg["confirmation_seeds"],
              "domain_count": len(domains), "development_record_count": len(development),
              "confirmation_record_count": len(confirmation),
              "temperature_by_budget_and_dimension": temperatures,
              "development_coverage": dev_coverage,
              "confirmation_coverage": confirmation_coverage,
              "confirmation_mean_absolute_coverage_error": confirmation_mace,
              "gate": {"maximum_mean_absolute_coverage_error":
                       cfg["maximum_mean_absolute_coverage_error"],
                       "passed": confirmation_mace <=
                       float(cfg["maximum_mean_absolute_coverage_error"])}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["gate"] | {
        "confirmation_mean_absolute_coverage_error": confirmation_mace}, indent=2))


if __name__ == "__main__":
    main()
