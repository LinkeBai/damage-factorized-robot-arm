import numpy as np

from scripts.diagnose_ipwm_action_ranking import rankdata, spearman


def test_rankdata_uses_average_rank_for_ties():
    actual = rankdata(np.array([3.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(actual, [3.0, 0.5, 0.5, 2.0])


def test_spearman_detects_order_and_reversal():
    values = np.array([0.2, 0.4, 0.8, 1.6])
    assert np.isclose(spearman(values, values), 1.0)
    assert np.isclose(spearman(values, values[::-1]), -1.0)


def test_spearman_returns_zero_for_unidentifiable_ranking():
    assert spearman(np.ones(4), np.arange(4.0)) == 0.0
