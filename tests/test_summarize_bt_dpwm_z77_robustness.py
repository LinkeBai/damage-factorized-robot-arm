import numpy as np

from scripts.summarize_bt_dpwm_z77_robustness import bootstrap_ci


def test_seed_bootstrap_interval_preserves_two_seed_extrema():
    interval = bootstrap_ci([1.5, 9.0], 10000, np.random.default_rng(7))
    assert interval == [1.5, 9.0]
