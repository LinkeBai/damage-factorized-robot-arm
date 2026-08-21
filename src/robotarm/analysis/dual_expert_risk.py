"""Fixed-depth risk metrics for dual-expert world models."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def selective_aurc(
    score: np.ndarray, error: np.ndarray, coverages: list[float]
) -> tuple[float, list[dict[str, float]]]:
    """Area under the selective RMSE-versus-coverage curve."""
    order = np.argsort(np.asarray(score))
    squared = np.asarray(error, dtype=float) ** 2
    curve = []
    for coverage in coverages:
        keep = max(1, int(np.ceil(len(order) * coverage)))
        rmse = float(np.sqrt(np.mean(squared[order[:keep]])))
        curve.append({"coverage": float(coverage), "rmse": rmse, "n": keep})
    x = np.asarray([item["coverage"] for item in curve])
    y = np.asarray([item["rmse"] for item in curve])
    return float(np.trapezoid(y, x) / (x[-1] - x[0])), curve


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Rank correlation of x/y residuals after linear rank control."""
    xr, yr, cr = percentile_rank(x), percentile_rank(y), percentile_rank(control)
    design = np.column_stack((np.ones_like(cr), cr))
    x_residual = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_residual = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if np.std(x_residual) < 1e-12 or np.std(y_residual) < 1e-12:
        return 0.0
    return float(spearmanr(x_residual, y_residual).statistic)


def fixed_depth_risk_summary(
    records: list[dict[str, float]], coverages: list[float]
) -> dict:
    depth_rows = []
    for depth in sorted({int(item["depth"]) for item in records}):
        subset = [item for item in records if int(item["depth"]) == depth]
        epistemic = np.asarray([item["object_epistemic"] for item in subset])
        cross = np.asarray([item["cross"] for item in subset])
        error = np.asarray([item["error"] for item in subset])
        epi_rank, cross_rank = percentile_rank(epistemic), percentile_rank(cross)
        combined = 0.5 * (epi_rank + cross_rank)
        baseline_aurc, baseline_curve = selective_aurc(epi_rank, error, coverages)
        combined_aurc, combined_curve = selective_aurc(combined, error, coverages)
        improvement = 100.0 * (baseline_aurc - combined_aurc) / baseline_aurc
        depth_rows.append({
            "depth": depth + 1,
            "n": len(subset),
            "baseline_aurc": baseline_aurc,
            "combined_aurc": combined_aurc,
            "aurc_improvement_pct": improvement,
            "partial_spearman": partial_spearman(cross, error, epistemic),
            "baseline_curve": baseline_curve,
            "combined_curve": combined_curve,
        })
    return {
        "mean_baseline_aurc": float(np.mean([row["baseline_aurc"] for row in depth_rows])),
        "mean_combined_aurc": float(np.mean([row["combined_aurc"] for row in depth_rows])),
        "mean_aurc_improvement_pct": float(np.mean([row["aurc_improvement_pct"] for row in depth_rows])),
        "mean_partial_spearman": float(np.mean([row["partial_spearman"] for row in depth_rows])),
        "depth_rows": depth_rows,
    }
