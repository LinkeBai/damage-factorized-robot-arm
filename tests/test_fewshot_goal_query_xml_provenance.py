import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bt_dpwm_fewshot_z48.py"


def test_every_cpu_goal_collection_receives_explicit_xml_path():
    """Prevent calibrated-arm queries silently falling back to arm_push.xml."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect_push_domains"
    ]
    assert calls
    for call in calls:
        assert "xml_path" in {keyword.arg for keyword in call.keywords}


def test_goal_query_cache_is_namespaced_by_xml_and_legacy_reuse_is_guarded():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_x{xml_fingerprint}_q{query_count}" in source
    assert "_x{xml_fingerprint}_v{validation_count}" in source
    assert "if resolved_xml == default_xml else None" in source
