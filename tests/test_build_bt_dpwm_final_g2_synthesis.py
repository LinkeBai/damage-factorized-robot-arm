import importlib.util
from pathlib import Path


def test_synthesis_contains_passes_and_failures():
    path = Path(__file__).parents[1] / "scripts" / "build_bt_dpwm_final_g2_synthesis.py"
    spec = importlib.util.spec_from_file_location("g2_synthesis", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.build(module.load_sources())
    statuses = {row["status"] for row in result["claims"]}
    assert {"PASS", "FAIL", "NOT SUPPORTED"} <= statuses
    assert result["verdict"]["g2_artifact_delivery_complete"] is True
