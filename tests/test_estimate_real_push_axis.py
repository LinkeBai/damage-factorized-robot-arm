import numpy as np

from scripts.estimate_real_push_axis import fit_push_axis


def test_push_axis_fit_passes_well_conditioned_line():
    samples = []
    for u in np.linspace(1000, 1060, 13):
        samples.append({"push_face_px": [u, 500], "tcp_base_xy_m": [0.1 + u * .001, .2 + u * .0002]})
    diagnostics, errors = fit_push_axis(samples)
    assert not errors
    assert diagnostics["pixel_x_span"] == 60
    assert diagnostics["fit_rmse_m"] < 1e-10


def test_push_axis_fit_rejects_short_coverage():
    samples = [
        {"push_face_px": [1000 + i, 500], "tcp_base_xy_m": [.2 + i * .001, .1]}
        for i in range(6)
    ]
    _, errors = fit_push_axis(samples)
    assert "pixel_axis_coverage_below_20px" in errors
