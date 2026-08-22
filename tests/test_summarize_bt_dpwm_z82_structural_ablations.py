import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "summarize_bt_dpwm_z82_structural_ablations.py"
    spec = importlib.util.spec_from_file_location("z82_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_ablation_names_are_frozen():
    module = _module()
    assert set(module.VARIANTS) == {
        "z82_no_analytic_projection", "z83_no_locked_residual_projection",
        "z84_post_object_residual", "z85_nonzero_k0"}
    assert "violation_rmse" in module.METRICS
