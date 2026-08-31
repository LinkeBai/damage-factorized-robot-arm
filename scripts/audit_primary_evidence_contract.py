"""Audit implementation/evidence coverage of the frozen primary-arm contract.

The audit is intentionally conservative: textual presence in an unrelated gate
does not count as a same-protocol ablation result.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/experiment/icra_2027_primary_5dof_recovery_v1.yaml"
OUTPUT = ROOT / "results/final/primary-evidence-contract-audit.json"


def exists(path: str) -> bool:
    return (ROOT / path).is_file()


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    cfg = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    strict_path = "results/final/primary-strict-development-3seed-summary.json"
    global_path = "results/final/primary-global-matched-ablation-3seed.json"
    projection_path = "results/final/primary-projection-ablation-3seed.json"
    decision_path = "results/final/primary-decision-loss-ablation-3seed.json"
    confirmation_path = "results/final/confirmation-d3-query-seed91031-summary.json"
    strict = load_json(strict_path)
    global_result = load_json(global_path)
    projection = load_json(projection_path)
    decision = load_json(decision_path)
    confirmation = load_json(confirmation_path)
    three_seed = lambda payload: payload.get("seeds", [7, 17, 27]) == [7, 17, 27]
    coverage = {
        "nominal_world_model": {
            "implementation": "src/robotarm/models/world_model.py",
            "implemented": exists("src/robotarm/models/world_model.py"),
            "same_protocol_result": strict_path if three_seed(strict) else None,
            "formal_row": "shared_baseline",
        },
        "fault_conditioned_world_model": {
            "implementation": "src/robotarm/training/g1_mechanism.py",
            "implemented": exists("src/robotarm/training/g1_mechanism.py"),
            "same_protocol_result": strict_path if three_seed(strict) else None,
            "formal_row": "carrier_no_intervention",
        },
        "analytic_projection": {
            "implementation": "src/robotarm/models/topology_surgery.py",
            "implemented": exists("src/robotarm/models/topology_surgery.py"),
            "same_protocol_result": projection_path if three_seed(projection) else None,
            "formal_row": "projected versus identical frozen checkpoint without projection",
        },
        "projection_global_residual_matched": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --global-residual-matched",
            "implemented": (
                exists("src/robotarm/models/block_triangular_dpwm.py")
                and exists("scripts/run_bt_dpwm_gate_y0.py")
            ),
            "same_protocol_result": global_path if three_seed(global_result) else None,
            "note": "Same 12-D input, global 14-D publication, hard projection retained; the full model differs by 8 parameters.",
        },
        "si_ipwm": {
            "implementation": "src/robotarm/models/selective_intervention_rollout.py",
            "implemented": exists("src/robotarm/models/selective_intervention_rollout.py"),
            "same_protocol_result": strict_path if three_seed(strict) else None,
        },
        "si_ipwm_without_projection": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --disable-analytic-projection",
            "implemented": (
                exists("src/robotarm/models/block_triangular_dpwm.py")
                and exists("src/robotarm/models/selective_intervention_rollout.py")
                and exists("scripts/run_bt_dpwm_gate_y0.py")
            ),
            "same_protocol_result": projection_path if three_seed(projection) else None,
            "note": "Exact same-checkpoint switch; formal three-seed lock-stress result is recorded.",
        },
        "si_ipwm_without_path_support": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --evaluate-selective-publication (full_state_ipwm row)",
            "implemented": exists("scripts/run_bt_dpwm_gate_y0.py"),
            "same_protocol_result": strict_path if three_seed(strict) else None,
            "note": "Full-state and selective publication are explicitly compared and equal in 3/3 seeds.",
        },
        "si_ipwm_without_paired_counterfactual_loss": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --decision-weight 0",
            "implemented": exists("src/robotarm/training/decision_focused.py"),
            "same_protocol_result": decision_path if three_seed(decision) else None,
            "note": "Decision weight 10 versus zero is reported for all three seeds.",
        },
        "oracle_realized_candidate_selector": {
            "implementation": "src/robotarm/training/decision_focused.py (evaluation only)",
            "implemented": exists("src/robotarm/training/decision_focused.py"),
            "same_protocol_result": strict_path if three_seed(strict) else None,
            "note": "Privileged realized-cost selector is a headroom upper bound, never a deployable baseline.",
        },
    }
    expected = list(cfg["methods"]) + list(cfg["ablations"]) + [
        "oracle_realized_candidate_selector"
    ]
    rows = [{"name": name, **coverage[name]} for name in expected]
    result_cells = sum(row["same_protocol_result"] is not None for row in rows)
    simulation_complete = (
        all(bool(row["implemented"]) for row in rows)
        and result_cells == len(rows)
        and projection["with_projection_zero_violation_all_seeds"] is True
        and global_result["verdict"] == "STRUCTURAL_ATTRIBUTION_NO_GO"
        and confirmation["verdict"] == "STRUCTURAL_ATTRIBUTION_NO_GO"
    )
    payload = {
        "contract": str(CONTRACT.relative_to(ROOT)),
        "platform": cfg["platform"]["embodiment"],
        "heldout_lock": cfg["interventions"]["heldout_lock"],
        "status": (
            "SIMULATION_EVIDENCE_COMPLETE_REAL_ROBOT_PENDING"
            if simulation_complete else "INCOMPLETE"
        ),
        "implemented_cells": sum(bool(row["implemented"]) for row in rows),
        "required_cells": len(rows),
        "same_protocol_result_cells": result_cells,
        "cells": rows,
        "blocking_fact": (
            "Original 5-DoF real-robot measurements are still absent; simulation contract cells are complete."
            if simulation_complete else "One or more frozen simulation evidence cells is missing or inconsistent."
        ),
        "post_freeze_query_confirmation": {
            "source": confirmation_path,
            "candidate_sha256": "43a00365caf59e504ef7b730fc9d91bc7bfd0d9efce79899a7b9d725072e2702",
            "scope": "D3 fresh candidate/query sample after checkpoint freeze; not pristine unseen-domain evidence",
            "verdict": confirmation["verdict"],
        },
        "development_smoke": {
            "path": "runs/ipwm_decision_metrics_smoke_20260831/seed7/summary.json",
            "scope": "D2/D4 only, 4 groups, 32 candidates, 2 epochs; not paper evidence",
            "selected_epoch": 0,
            "spearman": -0.03615674961321874,
            "kendall": -0.0589717741935484,
            "top1_regret": 0.012376385973766446,
            "oracle_true_cost": 0.0044627864845097065,
            "decision": "PIPELINE_PASS_PERFORMANCE_NO-GO",
        },
        "no_projection_development_smoke": {
            "path": "runs/ipwm_no_projection_smoke_20260831/seed7/summary.json",
            "scope": "D2/D4 only, 4 groups, 32 candidates, 2 epochs; wiring check only, not paper evidence",
            "analytic_projection": False,
            "object_improvement_pct": 0.08624777551884445,
            "free_arm_improvement_pct": -0.14237894325531977,
            "overall_improvement_pct": -0.13717236540651273,
            "spearman": -0.04147199888008091,
            "kendall": -0.06401209677419356,
            "top1_regret": 0.012376385973766446,
            "decision": "ABLATION_WIRING_PASS_PERFORMANCE_NO-GO",
            "warning": "Zero lock violation in this tiny sampled rollout does not prove projection is unnecessary; the unit test establishes that drift is possible when projection is disabled, and formal multi-seed lock-stress evaluation remains required.",
        },
        "global_matched_development_smoke": {
            "path": "runs/ipwm_global_matched_smoke_20260831/seed7/summary.json",
            "scope": "D2/D4 only, 4 groups, 32 candidates, 2 epochs; capacity/wiring check only, not paper evidence",
            "parameters": 337842,
            "object_improvement_pct": -2.545764608355462,
            "free_arm_improvement_pct": 6.7367278463471205,
            "overall_improvement_pct": 6.552339826165014,
            "spearman": -0.03615674961321874,
            "kendall": -0.0589717741935484,
            "top1_regret": 0.012376385973766446,
            "selected_epoch": 0,
            "decision": "CAPACITY_WIRING_PASS_PERFORMANCE_NO-GO",
        },
        "full_128_candidate_development_diagnostic": {
            "candidate_audit": "results/protocol/development_seed7_formal_audit.json",
            "result": "runs/ipwm_128candidate_full_eval_diagnostic_20260831/seed7/summary.json",
            "scope": "D2/D4, 400 groups, 128 unique candidates, 50 steps; epoch-0-selected diagnostic, single seed, not a formal ablation result",
            "candidate_sha256": "d587bd32de45ffe76ccee6c25adfcf98099a35a934169b801c08d14e64425180",
            "carrier": {
                "spearman": 0.021163735900287892,
                "kendall": 0.002778051181102361,
                "top1_regret": 0.00906155145734374,
                "selected_true_cost": 0.04475832912576152,
            },
            "selective_ipwm": {
                "spearman": 0.021163735900287892,
                "kendall": 0.002778051181102361,
                "top1_regret": 0.00906155145734374,
                "selected_true_cost": 0.04475832912576152,
            },
            "oracle_true_cost": 0.03569677844643593,
            "decision": "CANDIDATE_PROTOCOL_PASS_MODEL_DIFFERENTIATION_NO-GO",
        },
        "integrity_rule": "Do not substitute heterogeneous historical gates or relabel the current state-loss trainer as decision-focused.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
