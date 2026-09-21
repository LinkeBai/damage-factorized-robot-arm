from scripts.generate_real_robot_level_a_schedule import build


def test_level_a_schedule_has_no_learned_method_labels() -> None:
    rows = build(20260901, 10, {
        "intact": "safe_intact_v1", "D2": "safe_d2_v1", "D3": "safe_d3_v1"})
    assert len(rows) == 60
    assert {row["condition"] for row in rows} == {"intact", "D2", "D3"}
    assert {row["method"] for row in rows} == {"fixed_safe_trajectory"}
    assert len({row["trial_order"] for row in rows}) == 60
    assert all(row["trajectory_id"] for row in rows)
    assert {row["position_id"] for row in rows} == {"A"}
    primary = [row for row in rows if row["trial_role"] == "primary"]
    reserves = [row for row in rows if row["trial_role"] == "reserve"]
    assert len(primary) == 30
    assert len(reserves) == 30
    assert max(int(row["trial_order"]) for row in primary) < min(
        int(row["trial_order"]) for row in reserves)
    for condition in ("intact", "D2", "D3"):
        assert sorted(
            int(row["reserve_rank"]) for row in reserves
            if row["condition"] == condition
        ) == list(range(1, 11))


def test_level_a_rejects_fewer_than_ten_repetitions() -> None:
    try:
        build(1, 5, {"intact": "a", "D2": "b", "D3": "c"})
    except ValueError as exc:
        assert "at least 10" in str(exc)
    else:
        raise AssertionError("unsafe undersized formal schedule was accepted")
