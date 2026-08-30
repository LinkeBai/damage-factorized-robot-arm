from scripts.calibrate_bt_dpwm_z81_conformal_context import fit_conformal_radii


def test_conformal_radius_uses_empirical_standardized_quantile():
    records = [
        {"budget": 3, "normalized_error": [value],
         "normalized_variance": [1.0]}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    radii = fit_conformal_radii(records, [3], [0.5])
    assert radii["3"]["0.5"] == [3.0]
