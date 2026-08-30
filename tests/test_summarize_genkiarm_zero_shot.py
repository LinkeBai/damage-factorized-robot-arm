import numpy as np

from scripts.summarize_genkiarm_zero_shot import bootstrap_interval


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.array([0.0, 0.0, 12.0])
    first = bootstrap_interval(values)
    second = bootstrap_interval(values)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
