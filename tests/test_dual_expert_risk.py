import numpy as np

from robotarm.analysis.dual_expert_risk import (
    fixed_depth_risk_summary,
    partial_spearman,
    percentile_rank,
)


def test_percentile_rank_handles_ties_and_endpoints():
    result = percentile_rank(np.array([3.0, 1.0, 3.0, 2.0]))
    assert np.allclose(result, [5 / 6, 0.0, 5 / 6, 1 / 3])


def test_partial_spearman_detects_signal_beyond_control():
    rng = np.random.default_rng(7)
    control = rng.normal(size=200)
    cross = rng.normal(size=200)
    error = control + 2.0 * cross + rng.normal(scale=0.1, size=200)
    assert partial_spearman(cross, error, control) > 0.8


def test_fixed_depth_combination_improves_when_cross_finds_hard_cases():
    records = []
    for depth in range(2):
        for index in range(40):
            records.append({
                "depth": depth,
                "object_epistemic": float(index % 5),
                "cross": float(index),
                "error": float(index + 1),
            })
    result = fixed_depth_risk_summary(records, [0.25, 0.5, 0.75, 1.0])
    assert result["mean_aurc_improvement_pct"] > 0
    assert result["mean_partial_spearman"] > 0.9
