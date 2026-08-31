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


def main() -> None:
    cfg = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    coverage = {
        "nominal_world_model": {
            "implementation": "src/robotarm/models/world_model.py",
            "implemented": exists("src/robotarm/models/world_model.py"),
            "same_protocol_result": None,
        },
        "fault_conditioned_world_model": {
            "implementation": "src/robotarm/training/g1_mechanism.py",
            "implemented": exists("src/robotarm/training/g1_mechanism.py"),
            "same_protocol_result": None,
        },
        "analytic_projection": {
            "implementation": "src/robotarm/models/topology_surgery.py",
            "implemented": exists("src/robotarm/models/topology_surgery.py"),
            "same_protocol_result": None,
        },
        "projection_global_residual_matched": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --global-residual-matched",
            "implemented": (
                exists("src/robotarm/models/block_triangular_dpwm.py")
                and exists("scripts/run_bt_dpwm_gate_y0.py")
            ),
            "same_protocol_result": None,
            "note": "Same 12-D input, global 14-D publication, hard projection retained; frozen ranks differ by 8 parameters over the full model. Formal run remains missing.",
        },
        "si_ipwm": {
            "implementation": "src/robotarm/models/selective_intervention_rollout.py",
            "implemented": exists("src/robotarm/models/selective_intervention_rollout.py"),
            "same_protocol_result": None,
        },
        "si_ipwm_without_projection": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --disable-analytic-projection",
            "implemented": (
                exists("src/robotarm/models/block_triangular_dpwm.py")
                and exists("src/robotarm/models/selective_intervention_rollout.py")
                and exists("scripts/run_bt_dpwm_gate_y0.py")
            ),
            "same_protocol_result": None,
            "note": "Exact same-capacity switch is implemented; formal same-protocol result remains missing. Existing Z82 cannot fill this cell.",
        },
        "si_ipwm_without_path_support": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --evaluate-selective-publication (full_state_ipwm row)",
            "implemented": exists("scripts/run_bt_dpwm_gate_y0.py"),
            "same_protocol_result": None,
        },
        "si_ipwm_without_paired_counterfactual_loss": {
            "implementation": "scripts/run_bt_dpwm_gate_y0.py --decision-weight 0",
            "implemented": exists("src/robotarm/training/decision_focused.py"),
            "same_protocol_result": None,
            "note": "The paired loss and zero-weight ablation are implemented; formal same-protocol runs remain missing.",
        },
    }
    expected = list(cfg["methods"]) + list(cfg["ablations"])
    rows = [{"name": name, **coverage[name]} for name in expected]
    payload = {
        "contract": str(CONTRACT.relative_to(ROOT)),
        "platform": cfg["platform"]["embodiment"],
        "heldout_lock": cfg["interventions"]["heldout_lock"],
        "status": "INCOMPLETE",
        "implemented_cells": sum(bool(row["implemented"]) for row in rows),
        "required_cells": len(rows),
        "same_protocol_result_cells": sum(row["same_protocol_result"] is not None for row in rows),
        "cells": rows,
        "blocking_fact": "The frozen contract is not an executable unified ablation yet.",
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
