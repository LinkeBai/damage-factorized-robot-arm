"""Validate and summarize paired original-arm Push trials without imputation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED = {
    "pair_id", "condition", "method", "position_id", "trial_order", "aborted",
    "max_lock_error_rad", "reached", "contact", "endpoint_error_m", "success",
    "camera_left_video", "camera_horizontal_video", "control_log", "failure_code",
}


def flag(value: str) -> int:
    if value.strip().lower() in {"1", "true", "yes"}: return 1
    if value.strip().lower() in {"0", "false", "no"}: return 0
    raise ValueError(f"Invalid Boolean value: {value!r}")


def interval(values: np.ndarray, seed: int = 20260901) -> list[float]:
    if len(values) < 2: return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(20_000)])
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/real_robot/push-summary.json"))
    parser.add_argument("--require-files", action="store_true",
                        help="Require both videos and the control log to exist on disk")
    parser.add_argument("--reference-method", default="nominal",
                        help="Reference method used in the paired comparison")
    parser.add_argument("--candidate-method", default="global_matched",
                        help="Candidate method used in the paired comparison")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing: raise SystemExit(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Trial table contains no rows")
    orders = [row["trial_order"].strip() for row in rows]
    if any(not value for value in orders) or len(set(orders)) != len(orders):
        raise SystemExit("trial_order must be present and unique for every row")
    required_valid_values = {
        "max_lock_error_rad", "reached", "contact", "endpoint_error_m", "success",
        "camera_left_video", "camera_horizontal_video", "control_log",
    }
    validation_errors: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            aborted = flag(row["aborted"])
        except ValueError as exc:
            validation_errors.append(f"line {line_number}: {exc}")
            continue
        if aborted:
            if not row["failure_code"].strip():
                validation_errors.append(
                    f"line {line_number}: aborted trial requires failure_code"
                )
            continue
        missing_values = sorted(name for name in required_valid_values if not row[name].strip())
        if missing_values:
            validation_errors.append(
                f"line {line_number}: non-aborted trial missing {missing_values}"
            )
            continue
        for name in ("reached", "contact", "success"):
            try:
                flag(row[name])
            except ValueError as exc:
                validation_errors.append(f"line {line_number}: {exc}")
        for name in ("max_lock_error_rad", "endpoint_error_m"):
            try:
                value = float(row[name])
                if not np.isfinite(value) or value < 0:
                    raise ValueError
            except ValueError:
                validation_errors.append(
                    f"line {line_number}: {name} must be a finite non-negative number"
                )
        if args.require_files:
            for name in ("camera_left_video", "camera_horizontal_video", "control_log"):
                if not Path(row[name]).is_file():
                    validation_errors.append(
                        f"line {line_number}: {name} does not exist: {row[name]}"
                    )
    if validation_errors:
        preview = "\n".join(validation_errors[:20])
        suffix = "" if len(validation_errors) <= 20 else f"\n... {len(validation_errors)-20} more"
        raise SystemExit(f"Trial validity gate failed:\n{preview}{suffix}")
    valid = [row for row in rows if not flag(row["aborted"])]
    duplicate_keys = Counter(
        (row["condition"], row["pair_id"], row["method"]) for row in rows
    )
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate condition/pair/method rows: {duplicates[:10]}")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in valid: grouped.setdefault(row["method"], []).append(row)
    methods = {}
    for name, selected in grouped.items():
        methods[name] = {
            "trials": len(selected),
            "mean_endpoint_error_m": float(np.mean([float(row["endpoint_error_m"]) for row in selected])),
            "success_rate": float(np.mean([flag(row["success"]) for row in selected])),
            "reach_rate": float(np.mean([flag(row["reached"]) for row in selected])),
            "contact_rate": float(np.mean([flag(row["contact"]) for row in selected])),
            "max_lock_error_rad": float(max(float(row["max_lock_error_rad"]) for row in selected)),
        }
    feasibility_by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in valid if row["condition"] == condition]
        aborted = [row for row in rows
                   if row["condition"] == condition and flag(row["aborted"])]
        feasibility_by_condition[condition] = {
            "scheduled_trials": len(selected) + len(aborted),
            "valid_trials": len(selected),
            "aborted_trials": len(aborted),
            "mean_endpoint_error_m": (
                float(np.mean([float(row["endpoint_error_m"]) for row in selected]))
                if selected else None
            ),
            "success_rate": (
                float(np.mean([flag(row["success"]) for row in selected]))
                if selected else None
            ),
            "reach_rate": (
                float(np.mean([flag(row["reached"]) for row in selected]))
                if selected else None
            ),
            "contact_rate": (
                float(np.mean([flag(row["contact"]) for row in selected]))
                if selected else None
            ),
            "max_lock_error_rad": (
                float(max(float(row["max_lock_error_rad"]) for row in selected))
                if selected else None
            ),
            "failure_codes": dict(Counter(
                row["failure_code"] or "none" for row in rows
                if row["condition"] == condition
            )),
        }
    all_indexed = {}
    for row in rows:
        all_indexed.setdefault((row["condition"], row["pair_id"]), {})[
            row["method"]
        ] = row
    indexed = {}
    for row in valid:
        indexed.setdefault((row["condition"], row["pair_id"]), {})[
            row["method"]
        ] = row
    pair_items = [
        (key, pair) for key, pair in indexed.items()
        if args.reference_method in pair and args.candidate_method in pair
    ]
    pairs = [pair for _, pair in pair_items]
    mismatched_positions = [
        (pair[args.reference_method]["pair_id"],
         pair[args.reference_method]["position_id"],
         pair[args.candidate_method]["position_id"])
        for pair in pairs
        if pair[args.reference_method]["position_id"]
        != pair[args.candidate_method]["position_id"]
    ]
    if mismatched_positions:
        raise SystemExit(f"Paired rows have mismatched reset positions: {mismatched_positions[:10]}")
    def paired_values(selected_pairs):
        endpoint = np.asarray([
            float(p[args.reference_method]["endpoint_error_m"])
            - float(p[args.candidate_method]["endpoint_error_m"])
            for p in selected_pairs
        ])
        success = np.asarray([
            flag(p[args.candidate_method]["success"])
            - flag(p[args.reference_method]["success"]) for p in selected_pairs
        ], dtype=float)
        reach = np.asarray([
            flag(p[args.candidate_method]["reached"])
            - flag(p[args.reference_method]["reached"]) for p in selected_pairs
        ], dtype=float)
        contact = np.asarray([
            flag(p[args.candidate_method]["contact"])
            - flag(p[args.reference_method]["contact"]) for p in selected_pairs
        ], dtype=float)
        return endpoint, success, reach, contact

    def paired_summary(selected_pairs):
        endpoint, success, reach, contact = paired_values(selected_pairs)
        reference_endpoint = np.asarray([
            float(p[args.reference_method]["endpoint_error_m"])
            for p in selected_pairs
        ])
        reference_success = np.asarray([
            flag(p[args.reference_method]["success"]) for p in selected_pairs
        ], dtype=float)
        candidate_success = np.asarray([
            flag(p[args.candidate_method]["success"]) for p in selected_pairs
        ], dtype=float)
        def summarize(values):
            return {
                "mean": float(values.mean()) if len(values) else None,
                "ci95": interval(values) if len(values) else None,
            }
        reference_failure_rate = (
            float(1.0 - reference_success.mean()) if len(reference_success) else None
        )
        candidate_failure_rate = (
            float(1.0 - candidate_success.mean()) if len(candidate_success) else None
        )
        relative_failure_reduction = None
        if reference_failure_rate is not None and reference_failure_rate > 0:
            relative_failure_reduction = float(
                (reference_failure_rate - candidate_failure_rate)
                / reference_failure_rate
            )
        relative_endpoint_reduction = None
        if len(reference_endpoint) and float(reference_endpoint.mean()) > 0:
            relative_endpoint_reduction = float(
                endpoint.mean() / reference_endpoint.mean()
            )
        return {
            "pairs": len(selected_pairs),
            "endpoint_improvement_m": summarize(endpoint),
            "relative_endpoint_error_reduction": relative_endpoint_reduction,
            "success_improvement": summarize(success),
            "reference_failure_rate": reference_failure_rate,
            "candidate_failure_rate": candidate_failure_rate,
            "relative_failure_rate_reduction": relative_failure_reduction,
            "discordant_success_pairs": {
                "candidate_rescues_reference_failure": int(np.sum(
                    (reference_success == 0) & (candidate_success == 1))),
                "candidate_breaks_reference_success": int(np.sum(
                    (reference_success == 1) & (candidate_success == 0))),
            },
            "reach_improvement": summarize(reach),
            "contact_improvement": summarize(contact),
        }

    endpoint, success, reach, contact = paired_values(pairs)
    overall_pair_summary = paired_summary(pairs)
    condition_pairs = Counter(
        p[args.reference_method]["condition"] for p in pairs
    )
    per_condition = {
        condition: paired_summary([
            pair for key, pair in pair_items if key[0] == condition
        ])
        for condition in sorted({key[0] for key in all_indexed})
    }
    required_methods = {args.reference_method, args.candidate_method}
    incomplete_pair_keys = [
        {"condition": condition, "pair_id": pair_id,
         "present_methods": sorted(pair),
         "aborted_methods": sorted(
             method for method, row in pair.items() if flag(row["aborted"])
         )}
        for (condition, pair_id), pair in sorted(all_indexed.items())
        if not required_methods.issubset(indexed.get((condition, pair_id), {}))
    ]
    per_pair = [{
        "condition": key[0], "pair_id": key[1],
        "position_id": pair[args.reference_method]["position_id"],
        "endpoint_improvement_m": float(pair[args.reference_method]["endpoint_error_m"])
        - float(pair[args.candidate_method]["endpoint_error_m"]),
        "success_improvement": flag(pair[args.candidate_method]["success"])
        - flag(pair[args.reference_method]["success"]),
        "reach_improvement": flag(pair[args.candidate_method]["reached"])
        - flag(pair[args.reference_method]["reached"]),
        "contact_improvement": flag(pair[args.candidate_method]["contact"])
        - flag(pair[args.reference_method]["contact"]),
    } for key, pair in pair_items]
    formal_counts_met = all(
        per_condition.get(condition, {}).get("pairs", 0) >= 10
        for condition in ("D2", "D3")
    )
    feasibility_counts_met = all(
        feasibility_by_condition.get(condition, {}).get("valid_trials", 0) >= 10
        for condition in ("intact", "D2", "D3")
    )
    payload = {
        "source": str(args.csv), "rows": len(rows), "valid_rows": len(valid),
        "aborted_rows": len(rows) - len(valid), "methods": methods, "paired_trials": len(pairs),
        "paired_comparison": {
            "reference_method": args.reference_method,
            "candidate_method": args.candidate_method,
            "pairs_by_condition": dict(condition_pairs),
            "incomplete_or_aborted_pairs": incomplete_pair_keys,
        },
        "paired_endpoint_improvement_m": {"mean": float(endpoint.mean()) if len(endpoint) else None, "ci95": interval(endpoint) if len(endpoint) else None},
        "paired_relative_endpoint_error_reduction": overall_pair_summary[
            "relative_endpoint_error_reduction"],
        "paired_success_improvement": {"mean": float(success.mean()) if len(success) else None, "ci95": interval(success) if len(success) else None},
        "paired_failure_analysis": {
            "reference_failure_rate": overall_pair_summary["reference_failure_rate"],
            "candidate_failure_rate": overall_pair_summary["candidate_failure_rate"],
            "relative_failure_rate_reduction": overall_pair_summary[
                "relative_failure_rate_reduction"],
            "discordant_success_pairs": overall_pair_summary[
                "discordant_success_pairs"],
            "interpretation_rule": (
                "Descriptive effect size under the frozen 30 mm success threshold; "
                "always report with absolute success difference, paired CI, and counts."
            ),
        },
        "paired_reach_improvement": {"mean": float(reach.mean()) if len(reach) else None, "ci95": interval(reach) if len(reach) else None},
        "paired_contact_improvement": {"mean": float(contact.mean()) if len(contact) else None, "ci95": interval(contact) if len(contact) else None},
        "paired_by_condition": per_condition,
        "paired_rows": per_pair,
        "failure_codes": dict(Counter(row["failure_code"] or "none" for row in rows)),
        "conditions": dict(Counter(row["condition"] for row in valid)),
        "physical_feasibility_by_condition": feasibility_by_condition,
        "physical_feasibility_claim_level": (
            "formal" if feasibility_counts_met and args.require_files
            else "pilot" if valid else "no physical evidence"
        ),
        "physical_feasibility_gate": {
            "minimum_valid_trials_each_intact_D2_D3": 10,
            "counts_met": feasibility_counts_met,
            "raw_files_required_and_checked": args.require_files,
            "claim_boundary": (
                "Supports physical constraint/reach/contact/Push feasibility only; "
                "does not establish learned-method superiority."
            ),
        },
        "all_required_files_checked": args.require_files,
        "claim_level": (
            "formal" if formal_counts_met and args.require_files
            else "pilot" if pairs else "no paired evidence"
        ),
        "formal_gate": {
            "minimum_complete_pairs_each_D2_D3": 10,
            "counts_met": formal_counts_met,
            "raw_files_required_and_checked": args.require_files,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
