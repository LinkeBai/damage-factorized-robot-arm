import importlib.util
from pathlib import Path


def test_current_g2_evidence_package_passes_completion_audit():
    path = Path(__file__).parents[1] / "scripts" / "audit_bt_dpwm_g2_evidence.py"
    spec = importlib.util.spec_from_file_location("g2_audit", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.audit()
    assert result["passed"], [row for row in result["checks"] if not row["passed"]]
