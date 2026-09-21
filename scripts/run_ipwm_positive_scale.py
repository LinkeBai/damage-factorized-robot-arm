"""Frozen, resumable large-scale replication of the historical IPWM advantage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from scripts import audit_ipwm_targeted_advantage as audit
from scripts import replicate_ipwm_legacy_policy as legacy
from scripts import run_push_benchmark as collector
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors

OUT = ROOT / "runs/ipwm_positive_scale_20260911"
PROTOCOL = OUT / "protocol.json"
PHYSICS = audit.PHYSICS
SEEDS = [27, 37, 47]
SHARDS = 12
TRAJECTORIES_PER_SHARD = 100


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def source_paths() -> list[Path]:
    paths = [Path(__file__), ROOT / "scripts/audit_ipwm_targeted_advantage.py",
             ROOT / "scripts/replicate_ipwm_legacy_policy.py",
             ROOT / "scripts/run_push_benchmark.py", ROOT / "sim/assets/arm_push.xml",
             ROOT / "config/experiment/g2_ipwm_d3_physics_spectrum_seed27_audit_v1.yaml"]
    for seed in SEEDS:
        paths.extend([
            ROOT / ("runs/g2_r0_physical_context_residual/seed27_confirmation_v1/model.pt" if seed == 27 else
                    f"runs/g2_r0_physical_context_residual_extension/seed{seed}_v1/model.pt"),
            ROOT / f"runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_v1/bt_adapter.pt",
            ROOT / f"runs/g2_bt_dpwm_context_encoder_z65/seed{seed}_v1/context_encoder.pt",
        ])
    return paths


def freeze() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    spec = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Large independent-trajectory replication of the pre-existing historical selective-IPWM advantage.",
        "model_family": "historical selective IPWM and its mechanism-matched carrier",
        "seeds": SEEDS, "topology": "D3", "physics": PHYSICS,
        "primary_stratum": ["high_damping", "mixed_composition", "mixed_unseen"],
        "shards_per_condition": SHARDS, "trajectories_per_shard": TRAJECTORIES_PER_SHARD,
        "independent_trajectories_per_condition": SHARDS * TRAJECTORIES_PER_SHARD,
        "independent_trajectories_total": len(PHYSICS) * SHARDS * TRAJECTORIES_PER_SHARD,
        "steps": 150, "horizons": [10, 25, 50],
        "collection_policy": "legacy fixed-side approach, original evaluation targets and goal excitation std 0.08",
        "primary_metric": "historical mixed-unit object-state RMSE, retained only for exact historical replication",
        "required_separate_metrics": ["object_xy_rmse", "object_velocity_rmse", "free_robot_rmse",
                                      "robot_carrier_max", "lock_violation"],
        "counting": "Only independently reset trajectories count; windows, methods and model seeds do not multiply sample count.",
        "inference": "All seeds, conditions, shards and horizons are retained. No stopping or selection based on performance.",
        "scope": "Fixed-side contact-neighborhood D3 prediction evidence; not new-task, object-diversity or task-success evidence.",
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
        "prior_evidence": {
            "query57_all_horizon_mean_improvement_pct": 16.74089847543534,
            "query57_positive_cells": "27/27",
            "fresh_100_h50_mean_improvement_pct": 21.07,
            "fresh_100_h50_positive_cells": "9/9"
        }
    }
    atomic_json(PROTOCOL, spec)


def verify_frozen() -> dict:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for relative, expected in spec["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dependency changed: {relative}")
    return spec


def collect(physics: str, shard: int) -> None:
    verify_frozen()
    destination = OUT / "data" / physics / f"shard-{shard:02d}.pt"
    complete = destination.with_suffix(".json")
    if complete.exists():
        record = json.loads(complete.read_text())
        if destination.is_file() and digest(destination) == record["sha256"]:
            return
        raise RuntimeError(f"Changed completed shard: {destination}")
    domain = audit.DomainSpec("D3", physics, "test")
    targets = load_target_split(ROOT / "config/splits/push_targets_5dof_v1.yaml")
    seed = 912000000 + PHYSICS.index(physics) * 100000 + shard * TRAJECTORIES_PER_SHARD
    trajectories = collector.collect_push_domains(
        (domain,), trajectories_per_domain=TRAJECTORIES_PER_SHARD, steps=150, seed=seed,
        targets=tuple(t.as_array() for t in targets.evaluation), excitation="goal", goal_exploration_std=.08,
        block_initial_xy=np.asarray([.24, .10]), xml_path=ROOT / "sim/assets/arm_push.xml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.pt")
    torch.save(trajectories, temp); temp.replace(destination)
    reset_fingerprints = []
    for trajectory in trajectories:
        state = trajectory.states[0].detach().cpu().numpy().astype(np.float32)
        action = trajectory.actions.detach().cpu().numpy().astype(np.float32)
        reset_fingerprints.append(hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest())
    if len(set(reset_fingerprints)) != TRAJECTORIES_PER_SHARD:
        raise RuntimeError("Duplicate reset/action fingerprints within shard")
    atomic_json(complete, {"physics": physics, "shard": shard, "seed": seed,
                           "trajectories": len(trajectories), "sha256": digest(destination),
                           "fingerprints": reset_fingerprints})


def configure_historical(seed: int, device: torch.device):
    models, parts = audit.historical_models(seed, device)
    full, carrier, selective, encoder, cfg = parts
    calibration_domain = audit.DomainSpec("D3", "mixed_unseen", "test")
    calibration = audit.collect(calibration_domain, 91102, True)[0]
    mask, _ = _damage_tensors([calibration_domain.damage], device)
    with torch.no_grad():
        mean, _ = encoder(calibration.states[None].to(device), calibration.actions[None].to(device),
                          mask, return_uncertainty=True)
    context = mean[0] * float(cfg["context_posterior_scale"])
    full.set_intervention_context(context); carrier.set_intervention_context(context)
    selective.set_residual_context(context)
    return models, float(context.norm())


def evaluate(seed: int, physics: str, shard: int) -> None:
    verify_frozen()
    data_path = OUT / "data" / physics / f"shard-{shard:02d}.pt"
    data_record = json.loads(data_path.with_suffix(".json").read_text())
    if digest(data_path) != data_record["sha256"]:
        raise RuntimeError(f"Data hash mismatch: {data_path}")
    destination = OUT / "evaluation" / f"seed{seed}" / physics / f"shard-{shard:02d}.json"
    if destination.exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    models, context_norm = configure_historical(seed, device)
    trajectories = torch.load(data_path, weights_only=False)
    rows = audit.evaluate(models, trajectories, audit.DomainSpec("D3", physics, "test"), device)
    if not rows or not all(np.isfinite(list(r[k] for k in ["object_mse", "object_xy_mse",
                                                            "object_velocity_mse", "free_mse",
                                                            "robot_carrier_max", "lock_violation"])).all()
                           for r in rows):
        raise RuntimeError("Non-finite or empty evaluation")
    atomic_json(destination, {"seed": seed, "physics": physics, "shard": shard,
                              "data_sha256": data_record["sha256"], "context_norm": context_norm,
                              "device": str(device), "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--physics", choices=PHYSICS)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.collect: collect(args.physics, args.shard)
    elif args.evaluate: evaluate(args.seed, args.physics, args.shard)
    else: parser.error("Choose --freeze, --collect or --evaluate")


if __name__ == "__main__":
    main()
