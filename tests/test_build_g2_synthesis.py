from scripts.build_g2_synthesis import _coverage_item, build_synthesis


def test_coverage_lookup_uses_value_not_position() -> None:
    artifact = {
        "selective_prediction_curve": [
            {"coverage": 0.5, "mean_reduction_pct": 50.5},
            {"coverage": 0.7, "mean_reduction_pct": 31.0},
        ]
    }
    assert _coverage_item(artifact, 0.5)["mean_reduction_pct"] == 50.5
    assert _coverage_item(artifact, 0.7)["mean_reduction_pct"] == 31.0


def test_repository_synthesis_has_frozen_claim_boundaries() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data, rows = build_synthesis(root)
    assert data["seeds"] == [7, 17, 27, 37, 47]
    assert data["uncertainty"]["coverage_50"]["mean_reduction_pct"] > 50.0
    assert data["prediction"]["g2_structured_vs_ordinary"]["bootstrap_95_ci"][0] < 0
    assert any(row["status"] == "ci_crosses_zero" for row in rows)
