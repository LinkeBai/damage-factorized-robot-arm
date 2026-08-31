from scripts.audit_large_advantage_metrics import build


def test_large_advantage_audit_keeps_claim_boundaries() -> None:
    payload = build()
    rows = {row["metric"]: row for row in payload["candidates"]}
    projection = rows["locked-joint structural violation elimination"]
    regret = rows["top-1 action-regret reduction"]
    selective = rows["selective-prediction RMSE reduction at 50% coverage"]
    assert projection["advantage_percent"] == 100.0
    assert projection["evidence"]["with_projection_zero_violation_all_seeds"]
    assert regret["advantage_percent"] > 19.0
    assert regret["evidence"]["positive_seeds"] == 3
    assert selective["advantage_percent"] > 50.0
    assert selective["status"] == "SECONDARY_DEPTH_CONFOUNDED"
    assert "global residual" in payload["headline_pair"]["boundary"]
