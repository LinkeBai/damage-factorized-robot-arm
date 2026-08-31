import json
from pathlib import Path
import subprocess
import sys


def test_frozen_simulation_contract_has_all_formal_cells():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/audit_primary_evidence_contract.py"],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        (root / "results/final/primary-evidence-contract-audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "SIMULATION_EVIDENCE_COMPLETE_REAL_ROBOT_PENDING"
    assert payload["implemented_cells"] == payload["required_cells"] == 9
    assert payload["same_protocol_result_cells"] == payload["required_cells"]
    assert {cell["name"] for cell in payload["cells"]} >= {
        "nominal_world_model",
        "analytic_projection",
        "projection_global_residual_matched",
        "si_ipwm",
        "si_ipwm_without_projection",
        "si_ipwm_without_path_support",
        "si_ipwm_without_paired_counterfactual_loss",
        "oracle_realized_candidate_selector",
    }
