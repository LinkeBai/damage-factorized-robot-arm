"""Pair Z82--Z85 structural ablations with the frozen Z76 confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = {
    "z82_no_analytic_projection": "z82",
    "z83_no_locked_residual_projection": "z83",
    "z84_post_object_residual": "z84",
    "z85_nonzero_k0": "z85",
}
METRICS = ("overall_rmse", "free_rmse", "object_rmse", "violation_rmse")


def indexed_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(row["domain"], row["budget"]): row for row in payload["rows"]}


def summarize(root: Path, reference_root: Path, seeds):
    comparisons, failures = {}, []
    for name, folder in VARIANTS.items():
        records = []
        for seed in seeds:
            variant_path = root / folder / f"seed{seed}_v1" / "summary.json"
            reference_path = reference_root / f"seed{seed}_gate_v1" / "summary.json"
            if not variant_path.is_file() or not reference_path.is_file():
                failures.append({"variant": name, "seed": seed, "reason": "missing_summary"})
                continue
            variant, reference = indexed_rows(variant_path), indexed_rows(reference_path)
            if variant.keys() != reference.keys():
                failures.append({"variant": name, "seed": seed, "reason": "row_mismatch"})
                continue
            for key in sorted(variant):
                v, r = variant[key]["bt_dpwm"], reference[key]["bt_dpwm"]
                records.append({"seed": seed, "domain": key[0], "budget": key[1],
                    **{f"delta_{metric}": float(v[metric] - r[metric]) for metric in METRICS},
                    **{f"variant_{metric}": float(v[metric]) for metric in METRICS}})
        selected = records
        if name == "z85_nonzero_k0":
            selected = [row for row in records if row["budget"] == 0]
        elif name == "z84_post_object_residual":
            selected = [row for row in records if row["budget"] > 0]
        max_abs = {metric: (float(max(abs(row[f"delta_{metric}"]) for row in selected))
                            if selected else None) for metric in METRICS}
        comparisons[name] = {
            "record_count": len(records), "selected_record_count": len(selected),
            "max_abs_paired_delta": max_abs,
            "mean_paired_delta": {metric: (float(np.mean([
                row[f"delta_{metric}"] for row in selected])) if selected else None)
                for metric in METRICS},
            "maximum_variant_violation_rmse": (float(max(
                row["variant_violation_rmse"] for row in records)) if records else None),
            "records": records,
        }
    projection = comparisons["z82_no_analytic_projection"]
    locked = comparisons["z83_no_locked_residual_projection"]
    chain = comparisons["z84_post_object_residual"]
    k0 = comparisons["z85_nonzero_k0"]
    locked_delta = locked["max_abs_paired_delta"]["overall_rmse"]
    conclusions = {
        "analytic_projection_is_safety_necessary":
            (projection["maximum_variant_violation_rmse"] or 0.0) > 1e-7,
        "locked_residual_projection_is_deployment_redundant":
            locked_delta is not None and locked_delta <= 1e-12,
        "object_chain_has_measurable_effect":
            (chain["max_abs_paired_delta"]["object_rmse"] or 0.0) > 1e-9,
        "exact_k0_bypass_is_behaviorally_material":
            (k0["max_abs_paired_delta"]["overall_rmse"] or 0.0) > 1e-9,
    }
    return {"version": "g2_bt_dpwm_z82_z85_structural_ablation_summary_v1",
            "statistical_unit": "seed", "seeds": list(seeds), "failures": failures,
            "comparisons": comparisons, "conclusions": conclusions,
            "complete": not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "runs/g2_bt_dpwm_z82_structural_ablations"))
    parser.add_argument("--reference-root", type=Path, default=Path(
        "runs/g2_bt_dpwm_z76_confirmation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[57, 67])
    parser.add_argument("--output", type=Path, default=Path(
        "runs/g2_bt_dpwm_z82_structural_ablations/two_seed_summary_v1/summary.json"))
    args = parser.parse_args()
    output = summarize(args.root, args.reference_root, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"complete": output["complete"],
                      "conclusions": output["conclusions"]}, indent=2))
    return 0 if output["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
