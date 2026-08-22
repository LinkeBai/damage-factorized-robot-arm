from scripts.calibrate_bt_dpwm_z80_context_posterior import fit_temperature


def test_temperature_matches_normalized_squared_error_ratio():
    records = [
        {"budget": 3, "normalized_error": [2.0, 1.0],
         "normalized_variance": [1.0, 1.0]},
        {"budget": 3, "normalized_error": [0.0, 1.0],
         "normalized_variance": [1.0, 1.0]},
    ]
    temperature = fit_temperature(records, [3], [0.05, 20.0])["3"]
    assert temperature == [2.0, 1.0]
