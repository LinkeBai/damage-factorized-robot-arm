"""Evaluate a deployment-observable support-validation router for frozen IPWM.

The K25 calibration sequence is split causally: the first 20 transitions infer
physical context and the last five score full-intervention versus fallback.
The router accepts full intervention only when context is physically OOD, its
held-out support object error improves, and its held-out free-state error stays
within the pre-existing 5% engineering tolerance.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import (
    FewShotProjectedModel,
    ProjectedResidualInnovation,
)
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_push_benchmark import collect_push_domains


def build_strict(cfg: dict, device: torch.device) -> BlockTriangularDPWM:
    return BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=bool(cfg.get("contact_conditioned_robot", False)),
        contact_gated_object_context=bool(cfg.get("contact_gated_object_context", False)),
        independent_object_encoder=bool(cfg.get("independent_object_encoder", True)),
        object_hidden_dim=int(cfg.get("object_hidden_dim", cfg["hidden_dim"])),
        compact_bridge_object_head=bool(cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(cfg["geometric_object_rank"]),
        object_integration_dt=cfg.get("object_integration_dt"),
        object_position_blend=float(cfg.get("object_position_blend", 0.0)),
        geometric_object_contact_gate=bool(cfg.get("geometric_object_contact_gate", False)),
        intervention_residual_support_joints=tuple(
            int(x) for x in cfg.get("intervention_residual_support_joints", [])
        ),
        intervention_residual_meta_train=bool(
            cfg.get("intervention_residual_meta_train", False)
        ),
        intervention_object_rank=int(cfg.get("intervention_object_rank", 0)),
        object_bridge_alignment_rank=int(cfg.get("object_bridge_alignment_rank", 0)),
        intervention_residual_scale=float(cfg.get("intervention_residual_scale", 1.0)),
        intervention_residual_relative_clip=cfg.get("intervention_residual_relative_clip"),
        intervention_residual_decay=cfg.get("intervention_residual_decay"),
        intervention_context_dim=int(cfg.get("intervention_context_dim", 0)),
        intervention_context_rank=int(cfg.get("intervention_context_rank", 0)),
        intervention_context_strength=float(cfg.get("intervention_context_strength", 1.0)),
        intervention_context_ramp=float(cfg.get("intervention_context_ramp", 0.0)),
        intervention_context_ramp_start=int(cfg.get("intervention_context_ramp_start", 0)),
        intervention_context_delayed=bool(cfg.get("intervention_context_delayed", False)),
    ).to(device)


def make_adapter(cfg: dict, device: torch.device) -> ProjectedResidualInnovation:
    return ProjectedResidualInnovation(
        latent_dim=8,
        rank=int(cfg["adapter_rank"]),
        hidden_dim=int(cfg["adapter_hidden_dim"]),
        position_limit=float(cfg["correction_position_limit"]),
        velocity_limit=float(cfg["correction_velocity_limit"]),
        factorized_context=bool(cfg.get("factorized_context", False)),
        analytic_history=bool(cfg.get("analytic_history", False)),
        history_deadband=float(cfg.get("history_deadband", 0.04)),
    ).to(device)


@torch.no_grad()
def heldout_metrics(
    model: FewShotProjectedModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    angle: torch.Tensor,
    *,
    start: int,
) -> dict[str, float]:
    surgery = TopologySurgery()
    prediction = states[start : start + 1].clone()
    hidden = None
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    free_errors, object_errors = [], []
    for step in range(start, actions.shape[0]):
        raw, hidden = model.step(
            prediction, actions[step : step + 1], mask, angle, hidden
        )
        prediction = surgery.project_state(raw, mask, angle)
        error = (prediction - states[step + 1 : step + 2]).pow(2)
        free_errors.append((error[:, :10] * free_mask).sum(-1) / free_count)
        object_errors.append(error[:, 10:].mean(-1))
    return {
        "free_rmse": float(torch.cat(free_errors).mean().sqrt().cpu()),
        "object_rmse": float(torch.cat(object_errors).mean().sqrt().cpu()),
        "terminal_free_rmse": float(free_errors[-1].sqrt().cpu()),
        "terminal_object_rmse": float(object_errors[-1].sqrt().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-prefix", type=int, default=20)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--physical-threshold", type=float, default=1.2)
    parser.add_argument("--free-tolerance", type=float, default=0.05)
    parser.add_argument("--domains", nargs="+")
    args = parser.parse_args()
    if not 0 < args.context_prefix < args.budget:
        raise ValueError("context-prefix must lie strictly inside the support budget")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = build_strict(cfg, device)
    full.load_state_dict(torch.load(args.model, map_location=device))
    fallback = copy.deepcopy(full)
    with torch.no_grad():
        for name in ("geometric_object_head", "intervention_object_head"):
            if hasattr(fallback, name):
                for parameter in getattr(fallback, name).parameters():
                    parameter.zero_()

    adapter_cfg = yaml.safe_load(Path(cfg["matched_adapter_config"]).read_text(encoding="utf-8"))
    adapter_run = Path(str(cfg["matched_adapter_run_template"]).format(seed=args.seed))
    adapter_state = torch.load(adapter_run / "bt_adapter.pt", map_location=device)
    full_adapter, fallback_adapter = make_adapter(adapter_cfg, device), make_adapter(adapter_cfg, device)
    full_adapter.load_state_dict(adapter_state)
    fallback_adapter.load_state_dict(adapter_state)
    full_wrapped = FewShotProjectedModel(
        full, full_adapter, base_uses_topology=True
    ).to(device).eval()
    fallback_wrapped = FewShotProjectedModel(
        fallback, fallback_adapter, base_uses_topology=True
    ).to(device).eval()

    encoder = UncertainPhysicalContextEncoder(
        hidden_dim=int(cfg.get("context_encoder_hidden_dim", 96))
    ).to(device)
    encoder_path = Path(str(cfg["context_encoder_run_template"]).format(seed=args.seed)) / "context_encoder.pt"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    all_domains = tuple(dict.fromkeys((*protocol.train, *protocol.validation, *protocol.test)))
    requested = set(args.domains or [domain.domain_id for domain in protocol.test])
    rows = []
    for index, domain in enumerate(all_domains):
        if domain.domain_id not in requested:
            continue
        calibration_seed = args.seed * 100_000 + index * 1000 + 100
        trajectory = collect_push_domains(
            (domain,), trajectories_per_domain=1, steps=args.budget,
            seed=calibration_seed,
            targets=tuple(x.as_array() for x in targets.calibration),
            excitation="active",
            block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
            goal_exploration_std=float(q0a["goal_exploration_std"]),
        )[0]
        states, actions = trajectory.states.to(device), trajectory.actions.to(device)
        mask, angle = _damage_tensors([domain.damage], device)
        with torch.no_grad():
            mean, _ = encoder(
                states[: args.context_prefix + 1][None],
                actions[: args.context_prefix][None],
                mask,
                return_uncertainty=True,
            )
        context = mean[0] * float(cfg.get("context_posterior_scale", 1.0))
        full.set_intervention_context(context)
        fallback.set_intervention_context(context)
        full_wrapped.set_residual_context(context)
        fallback_wrapped.set_residual_context(context)
        full_metrics = heldout_metrics(
            full_wrapped, states, actions, mask, angle, start=args.context_prefix
        )
        fallback_metrics = heldout_metrics(
            fallback_wrapped, states, actions, mask, angle, start=args.context_prefix
        )
        context_norm = float(context.norm().cpu())
        physical_ood = context_norm >= args.physical_threshold
        object_improves = full_metrics["object_rmse"] < fallback_metrics["object_rmse"]
        free_safe = full_metrics["free_rmse"] <= (
            fallback_metrics["free_rmse"] * (1.0 + args.free_tolerance)
        )
        rows.append({
            "domain": domain.domain_id,
            "context_norm": context_norm,
            "physical_ood": physical_ood,
            "object_improves_on_holdout": object_improves,
            "free_safe_on_holdout": free_safe,
            "accepted": physical_ood and object_improves and free_safe,
            "full": full_metrics,
            "fallback": fallback_metrics,
            "holdout_object_improvement_pct": 100.0 * (
                fallback_metrics["object_rmse"] - full_metrics["object_rmse"]
            ) / fallback_metrics["object_rmse"],
            "holdout_free_change_pct": 100.0 * (
                full_metrics["free_rmse"] - fallback_metrics["free_rmse"]
            ) / fallback_metrics["free_rmse"],
        })

    output = {
        "version": "g2_ipwm_support_validation_gate_v1",
        "seed": args.seed,
        "context_prefix": args.context_prefix,
        "validation_suffix": args.budget - args.context_prefix,
        "physical_threshold": args.physical_threshold,
        "free_tolerance": args.free_tolerance,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
