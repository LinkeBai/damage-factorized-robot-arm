import math

import pytest

from scripts.audit_dual_camera_sync import (
    calculate_sync_metrics,
    nearest_midpoint_pairs,
    percentile,
)


def test_nearest_midpoint_pairing_uses_host_capture_midpoints() -> None:
    milliseconds = 1_000_000
    pairs = nearest_midpoint_pairs(
        [5 * milliseconds, 20 * milliseconds, 39 * milliseconds],
        [0, 10 * milliseconds, 30 * milliseconds, 50 * milliseconds],
    )

    # The first value is an exact tie and deterministically selects the earlier frame.
    assert pairs == [(0, 0, 5.0), (1, 1, 10.0), (2, 2, 9.0)]


def test_sync_metrics_report_sample_count_max_and_linear_p95() -> None:
    milliseconds = 1_000_000
    metrics = calculate_sync_metrics(
        [0, 10 * milliseconds, 20 * milliseconds, 30 * milliseconds],
        [0, 9 * milliseconds, 18 * milliseconds, 27 * milliseconds],
    )

    assert metrics["sample_count"] == 4
    assert metrics["maximum_observed_sync_error_ms"] == 3.0
    assert math.isclose(float(metrics["p95_sync_error_ms"]), 2.85)


def test_percentile_and_pairing_reject_invalid_samples() -> None:
    assert percentile([7.0], 95.0) == 7.0
    with pytest.raises(ValueError, match="at least one"):
        calculate_sync_metrics([], [1])
    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_sync_metrics([2, 1], [1, 2])
