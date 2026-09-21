"""Estimate the frozen horizontal Push axis from synchronized real evidence.

Unlike a 2-D hand-eye calibration, this estimator claims only the measured
one-dimensional task manifold.  It maps overhead image x to a base-frame XY
point on that line and fails closed when coverage or residuals are inadequate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from estimate_real_task_plane_affine import trial_samples


def fit_push_axis(samples: list[dict[str, object]]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if len(samples) < 6:
        return {"sample_count": len(samples)}, ["fewer_than_six_correspondences"]
    u = np.asarray([row["push_face_px"][0] for row in samples], dtype=np.float64)
    xy = np.asarray([row["tcp_base_xy_m"] for row in samples], dtype=np.float64)
    design = np.column_stack((u, np.ones(len(u))))
    coefficients, _, rank, singular = np.linalg.lstsq(design, xy, rcond=None)
    predicted = design @ coefficients
    residual_m = np.linalg.norm(predicted - xy, axis=1)
    slope = coefficients[0]
    metres_per_px = float(np.linalg.norm(slope))
    projection = (xy - coefficients[1]) @ slope / max(float(slope @ slope), 1e-15)
    pixel_residual = projection - u
    diagnostics = {
        "sample_count": len(samples),
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "pixel_x_span": float(np.ptp(u)),
        "base_axis_span_m": float(np.ptp(projection) * metres_per_px),
        "fit_rmse_m": float(np.sqrt(np.mean(residual_m ** 2))),
        "fit_p95_m": float(np.percentile(residual_m, 95)),
        "inverse_pixel_rmse_px": float(np.sqrt(np.mean(pixel_residual ** 2))),
        "base_xy_per_pixel": slope.tolist(),
        "base_xy_intercept_m": coefficients[1].tolist(),
        "metres_per_pixel": metres_per_px,
        "unit_axis_base_xy": (slope / max(metres_per_px, 1e-15)).tolist(),
    }
    if rank != 2:
        errors.append("task_axis_design_rank_is_not_two")
    if diagnostics["pixel_x_span"] < 20.0:
        errors.append("pixel_axis_coverage_below_20px")
    if diagnostics["base_axis_span_m"] < 0.03:
        errors.append("base_axis_coverage_below_30mm")
    if diagnostics["fit_rmse_m"] > 0.008:
        errors.append("base_fit_rmse_exceeds_8mm")
    if diagnostics["inverse_pixel_rmse_px"] > 5.0:
        errors.append("inverse_pixel_rmse_exceeds_5px")
    return diagnostics, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    samples, read_errors = [], []
    for trial in args.trials:
        try:
            samples.extend(trial_samples(trial.resolve(), args.stride))
        except Exception as exc:
            read_errors.append(f"{trial}: {type(exc).__name__}: {exc}")
    diagnostics, fit_errors = fit_push_axis(samples)
    errors = read_errors + fit_errors
    payload = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "scope": "frozen_overhead_horizontal_push_axis_only",
        "trials": sorted({str(row["trial_id"]) for row in samples}),
        "samples": samples,
        "diagnostics": diagnostics,
        "failure_reasons": errors,
        "gates": {
            "minimum_pixel_x_span": 20.0,
            "minimum_base_axis_span_m": 0.03,
            "maximum_base_fit_rmse_m": 0.008,
            "maximum_inverse_pixel_rmse_px": 5.0,
        },
        "claim_boundary": (
            "PASS supports conversion only on the frozen horizontal Push line; "
            "it is not a general 2-D hand-eye calibration or dynamics identification."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "trials", "diagnostics", "failure_reasons")}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
