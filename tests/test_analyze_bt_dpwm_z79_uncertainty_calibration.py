from scripts.analyze_bt_dpwm_z79_uncertainty_calibration import coverage_curve


def test_coverage_curve_retains_lowest_uncertainty_first():
    rows = [
        {"context_mean_std": 0.3, "candidate_harm_pct": -2.0},
        {"context_mean_std": 0.1, "candidate_harm_pct": 1.0},
        {"context_mean_std": 0.2, "candidate_harm_pct": 0.0},
        {"context_mean_std": 0.4, "candidate_harm_pct": -3.0},
    ]
    curve = coverage_curve(rows, [0.25, 0.5])
    assert curve[0]["retained"] == 1
    assert curve[0]["mean_candidate_harm_pct"] == 1.0
    assert curve[1]["retained"] == 2
