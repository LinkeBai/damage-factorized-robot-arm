import json
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
AUDIT = ROOT / "runs/g2_r0_icra_audit_20260824"


def test_manifest_artifacts_exist_and_match_sha256():
    summary = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["artifacts"]) == 42
    for artifact in summary["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert path.stat().st_size == artifact["bytes"], artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"], artifact["path"]


def test_five_seed_audit_is_complete_and_positive():
    summary = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["seeds"]) == {"7", "17", "27", "37", "47"}
    assert summary["gate"]["full_model_all_point_estimates_positive"]
    assert summary["gate"]["full_model_all_ci_lower_bounds_positive"]
    assert len(summary["cross_seed_summary"]) == 6
    assert all(row["seed_bootstrap_95ci_pct"][0] > 0
               for row in summary["cross_seed_summary"])


def test_raw_rows_reproduce_every_aggregate_rmse():
    for seed in (7, 17, 27, 37, 47):
        raw = json.loads((AUDIT / f"seed{seed}/raw_window_metrics_30traj.json")
                         .read_text(encoding="utf-8"))["rows"]
        aggregate = json.loads((AUDIT / f"seed{seed}/metrics_30traj.json")
                               .read_text(encoding="utf-8"))["rows"]
        assert len(raw) == 7200
        for expected in aggregate:
            selected = [row for row in raw
                        if row["domain"] == expected["domain"]
                        and row["horizon"] == expected["horizon"]
                        and row["method"] == expected["method"]]
            for short, raw_key in (
                ("free", "free_squared_error"),
                ("object", "object_squared_error"),
                ("overall", "overall_squared_error"),
                ("violation", "violation_squared_error"),
                ("pusher_xy", "pusher_xy_squared_error"),
            ):
                actual = float(np.sqrt(np.mean([row[raw_key] for row in selected])))
                assert np.isclose(actual, expected[f"{short}_rmse"], atol=1e-7)


def test_constraint_violation_is_exactly_zero_in_raw_rows():
    for seed in (7, 17, 27, 37, 47):
        rows = json.loads((AUDIT / f"seed{seed}/raw_window_metrics_30traj.json")
                          .read_text(encoding="utf-8"))["rows"]
        ours = [row for row in rows if row["method"] == "bt_matched_adapter"]
        assert ours
        assert max(row["violation_squared_error"] for row in ours) == 0.0
