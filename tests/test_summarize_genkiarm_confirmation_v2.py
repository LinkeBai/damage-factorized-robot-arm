from pathlib import Path

from scripts.summarize_genkiarm_confirmation_v2 import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_summary_uses_deployable_router_as_primary():
    paths = [ROOT / f"runs/g2_ipwm_selective_rollout_20260828/seed{s}/raw.json" for s in (27, 37, 47)]
    result = summarize(paths, draws=100, rng_seed=1)
    assert result["primary_method"] == "routed_selective_ipwm"
    assert result["seeds"] == [27, 37, 47]
    assert set(result["methods"]) == {"routed_selective_ipwm", "selective_ipwm"}
    assert 0 <= result["methods"]["routed_selective_ipwm"]["positive_seed_fraction"] <= 1
