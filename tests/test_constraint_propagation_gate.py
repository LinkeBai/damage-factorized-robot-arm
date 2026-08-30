import numpy as np

from scripts.fit_constraint_propagation_gate import base_features, structured_features


def test_structured_features_use_continuous_paths_for_unseen_lock():
    state = np.zeros((3, 14))
    action = np.zeros((3, 5))
    locks = np.array([0, 1, 2])
    base = base_features(state, action, locks)
    structured = structured_features(state, action, locks)
    assert base.shape == (3, 23)
    assert structured.shape == (3, 7, 32)
    # j3 has a continuous coordinate midway between training locks j2/j4.
    assert base[1, 19] == 0.5
    # The object node is downstream for every lock, but has a distinct type.
    assert np.all(structured[:, 0, 26] == 1.0)
    assert np.all(structured[:, 0, 27] == 1.0)
