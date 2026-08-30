"""Evaluate selective object-only IPWM against its mechanism-matched carrier."""
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

from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import FewShotProjectedModel
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.evaluate_g2_r0_core_metrics import evaluate
from scripts.evaluate_ipwm_support_validation_gate import build_strict, make_adapter
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_push_benchmark import collect_push_domains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--trajectories-per-domain", type=int, default=10)
    parser.add_argument("--context-budget", type=int, default=25)
    parser.add_argument("--physical-threshold", type=float, default=1.2)
    parser.add_argument(
        "--query-seed-base", type=int,
        help="Optional held-out trajectory/calibration seed, decoupled from the checkpoint seed.",
    )
    parser.add_argument("--domains", nargs="+")
    parser.add_argument(
        "--xml", type=Path, default=Path("sim/assets/arm_push.xml"),
        help="Frozen evaluation simulator; checkpoint and all learned components remain unchanged.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = build_strict(cfg, device)
    full.load_state_dict(torch.load(args.model, map_location=device))
    carrier = copy.deepcopy(full)
    with torch.no_grad():
        for name in ("geometric_object_head", "intervention_object_head"):
            if hasattr(carrier, name):
                for parameter in getattr(carrier, name).parameters():
                    parameter.zero_()

    adapter_cfg = yaml.safe_load(Path(cfg["matched_adapter_config"]).read_text(encoding="utf-8"))
    adapter_run = Path(str(cfg["matched_adapter_run_template"]).format(seed=args.seed))
    adapter_state = torch.load(adapter_run / "bt_adapter.pt", map_location=device)
    full_adapter, carrier_adapter = make_adapter(adapter_cfg, device), make_adapter(adapter_cfg, device)
    full_adapter.load_state_dict(adapter_state)
    carrier_adapter.load_state_dict(adapter_state)
    full_wrapped = FewShotProjectedModel(
        full, full_adapter, base_uses_topology=True
    ).to(device).eval()
    carrier_wrapped = FewShotProjectedModel(
        carrier, carrier_adapter, base_uses_topology=True
    ).to(device).eval()
    selective = SelectiveInterventionRollout(
        full_wrapped, carrier_wrapped
    ).to(device).eval()

    encoder = UncertainPhysicalContextEncoder(
        hidden_dim=int(cfg.get("context_encoder_hidden_dim", 96))
    ).to(device)
    encoder_path = Path(str(cfg["context_encoder_run_template"]).format(seed=args.seed)) / "context_encoder.pt"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    requested = set(args.domains or [domain.domain_id for domain in protocol.test])
    all_domains = tuple(dict.fromkeys((*protocol.train, *protocol.validation, *protocol.test)))
    rows, raw_rows, routing = [], [], []
    common = dict(
        steps=int(q0a["steps"]), excitation="goal",
        block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    for index, domain in enumerate(all_domains):
        if domain.domain_id not in requested:
            continue
        query_seed_base = args.seed if args.query_seed_base is None else args.query_seed_base
        test_seed = query_seed_base * 100_000 + index * 1000 + 500
        key = json.dumps({
            "kind": "push_test", "seed": test_seed,
            "domain": domain.domain_id, "q0a": q0a,
            "trajectories_per_domain": args.trajectories_per_domain,
            "xml": str(args.xml.resolve()),
        }, sort_keys=True)
        trajectories = cached_collect(
            args.cache_dir, key,
            lambda domain=domain: collect_push_domains(
                (domain,), trajectories_per_domain=args.trajectories_per_domain,
                seed=test_seed,
                targets=tuple(x.as_array() for x in targets.evaluation), **common,
                xml_path=args.xml,
            ),
        )
        calibration_seed = query_seed_base * 100_000 + index * 1000 + 100
        calibration_key = json.dumps({
            "kind": "push_context_calibration", "seed": calibration_seed,
            "domain": domain.domain_id, "budget": args.context_budget, "q0a": q0a,
            "xml": str(args.xml.resolve()),
        }, sort_keys=True)
        calibration = cached_collect(
            args.cache_dir, calibration_key,
            lambda domain=domain: collect_push_domains(
                (domain,), trajectories_per_domain=1, steps=args.context_budget,
                seed=calibration_seed,
                targets=tuple(x.as_array() for x in targets.calibration),
                excitation="active",
                block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
                goal_exploration_std=float(q0a["goal_exploration_std"]),
                xml_path=args.xml,
            ),
        )[0]
        mask, _ = _damage_tensors([domain.damage], device)
        with torch.no_grad():
            mean, _ = encoder(
                calibration.states[None].to(device),
                calibration.actions[None].to(device), mask,
                return_uncertainty=True,
            )
        context = mean[0] * float(cfg.get("context_posterior_scale", 1.0))
        full.set_intervention_context(context)
        carrier.set_intervention_context(context)
        selective.set_residual_context(context)
        context_norm = float(context.norm().cpu())
        routed = selective if context_norm >= args.physical_threshold else carrier_wrapped
        route_name = "selective_intervention" if routed is selective else "carrier_fallback"
        routing.append({
            "domain": domain.domain_id,
            "context_norm": context_norm,
            "route": route_name,
        })
        models = {
            "carrier_no_intervention": carrier_wrapped,
            "full_state_ipwm": full_wrapped,
            "selective_ipwm": selective,
            "routed_selective_ipwm": routed,
        }
        rows.extend(evaluate(
            models, trajectories, domain, (10, 25, 50), device, raw_rows=raw_rows
        ))

    output = {
        "version": "g2_ipwm_selective_rollout_v1",
        "seed": args.seed,
        "query_seed_base": args.seed if args.query_seed_base is None else args.query_seed_base,
        "physical_threshold": args.physical_threshold,
        "trajectories_per_domain": args.trajectories_per_domain,
        "xml": str(args.xml),
        "routing": routing,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps({
        "version": "g2_ipwm_selective_rollout_raw_v1",
        "seed": args.seed,
        "rows": raw_rows,
    }, indent=2), encoding="utf-8")
    print(f"[selective] wrote {len(rows)} aggregate rows and {len(raw_rows)} raw rows")


if __name__ == "__main__":
    main()
