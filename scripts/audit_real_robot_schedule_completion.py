"""Verify a completed real-robot trial log against its frozen schedule.

Legacy/Level-B schedules still require an exact row-for-row completion log.
Level-A reserve schedules require every primary row and permit only the frozen
per-condition reserve prefix needed to reach the preregistered valid-trial
target. Aborts remain in the log without enabling post-hoc retries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


BASE_IDENTITY_FIELDS = (
    "trial_order", "pair_id", "condition", "method", "position_id", "trajectory_id")
RESERVE_IDENTITY_FIELDS = ("trial_role", "reserve_rank")
SUCCESS_THRESHOLD_M = 0.03
MAXIMUM_LOCK_ERROR_RAD = math.radians(3.5)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_flag(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"invalid Boolean value {value!r}")


def audit(schedule: Path, completed: Path) -> dict:
    expected, observed = read(schedule), read(completed)
    errors: list[str] = []
    schedule_fields = set(expected[0]) if expected else set()
    completed_fields = set(observed[0]) if observed else set()
    has_any_reserve_field = bool(set(RESERVE_IDENTITY_FIELDS) & schedule_fields)
    reserve_protocol = set(RESERVE_IDENTITY_FIELDS).issubset(schedule_fields)
    if has_any_reserve_field and not reserve_protocol:
        errors.append("frozen schedule has an incomplete Level-A reserve schema")
    identity_fields = BASE_IDENTITY_FIELDS + (
        RESERVE_IDENTITY_FIELDS if reserve_protocol else ())
    required = set(identity_fields)
    if missing := required - schedule_fields:
        errors.append(f"frozen schedule missing identity fields: {sorted(missing)}")
    if missing := required - completed_fields:
        errors.append(f"completed log missing identity fields: {sorted(missing)}")

    expected_by_order = {row.get("trial_order", ""): row for row in expected}
    observed_by_order = {row.get("trial_order", ""): row for row in observed}
    if len(expected_by_order) != len(expected) or "" in expected_by_order:
        errors.append("frozen schedule has duplicate or blank trial_order")
    if len(observed_by_order) != len(observed) or "" in observed_by_order:
        errors.append("completed log has duplicate or blank trial_order")
    try:
        frozen_order = sorted(expected_by_order, key=int)
        completed_order = sorted(observed_by_order, key=int)
    except ValueError:
        frozen_order, completed_order = [], []
        errors.append("trial_order must be an integer in both files")

    expected_orders = set(expected_by_order)
    observed_orders = set(observed_by_order)
    extra_orders = sorted(observed_orders - expected_orders)
    if extra_orders:
        errors.append(f"completed log has unexpected trial orders: {extra_orders}")
    if reserve_protocol:
        primary_orders = {
            order for order, row in expected_by_order.items()
            if row.get("trial_role") == "primary"
        }
        missing_primary_orders = sorted(primary_orders - observed_orders)
        if missing_primary_orders:
            errors.append(
                "completed log removed frozen primary trial orders: "
                f"{missing_primary_orders}"
            )
    else:
        if len(expected) != len(observed):
            errors.append(
                f"row count differs: frozen={len(expected)}, completed={len(observed)}")
        missing_orders = sorted(expected_orders - observed_orders)
        if missing_orders:
            errors.append(f"completed log missing trial orders: {missing_orders}")

    changed = []
    for order in frozen_order:
        if order not in observed_by_order:
            continue
        before, after = expected_by_order[order], observed_by_order[order]
        differences = {
            field: {"frozen": before.get(field, ""), "completed": after.get(field, "")}
            for field in identity_fields if before.get(field, "") != after.get(field, "")
        }
        if differences:
            changed.append({"trial_order": order, "differences": differences})
    if changed:
        errors.append(f"{len(changed)} trial identity rows changed after freeze")

    parsed_aborted: dict[str, int] = {}
    inconsistent_success_rows = []
    lock_error_violations = []
    for order in completed_order:
        row = observed_by_order[order]
        try:
            aborted = parse_flag(row.get("aborted", ""))
            parsed_aborted[order] = aborted
        except ValueError as exc:
            errors.append(f"trial_order {order}: aborted {exc}")
            continue
        lock_text = row.get("max_lock_error_rad", "").strip()
        lock_error = None
        if lock_text:
            try:
                lock_error = float(lock_text)
                if not math.isfinite(lock_error) or lock_error < 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"trial_order {order}: max_lock_error_rad must be finite and non-negative")
                lock_error = None
        if lock_error is not None and lock_error > MAXIMUM_LOCK_ERROR_RAD + 1e-12:
            violation = {
                "trial_order": order,
                "condition": row.get("condition", ""),
                "aborted": bool(aborted),
                "max_lock_error_rad": lock_error,
                "maximum_allowed_lock_error_rad": MAXIMUM_LOCK_ERROR_RAD,
            }
            lock_error_violations.append(violation)
            if not aborted:
                errors.append(
                    f"trial_order {order}: non-aborted lock error {lock_error:.9g} rad "
                    f"exceeds 3.5 deg ({MAXIMUM_LOCK_ERROR_RAD:.9g} rad)")
        if aborted:
            if not row.get("failure_code", "").strip():
                errors.append(f"trial_order {order}: aborted trial requires failure_code")
            continue
        missing_outcomes = [
            name for name in ("max_lock_error_rad", "endpoint_error_m", "success")
            if not row.get(name, "").strip()
        ]
        if missing_outcomes:
            errors.append(
                f"trial_order {order}: non-aborted trial missing outcomes {missing_outcomes}")
            continue
        try:
            endpoint_error = float(row["endpoint_error_m"])
            if not math.isfinite(endpoint_error) or endpoint_error < 0:
                raise ValueError
        except ValueError:
            errors.append(
                f"trial_order {order}: endpoint_error_m must be finite and non-negative")
            continue
        try:
            success = parse_flag(row["success"])
        except ValueError as exc:
            errors.append(f"trial_order {order}: success {exc}")
            continue
        expected_success = int(endpoint_error <= SUCCESS_THRESHOLD_M)
        if success != expected_success:
            mismatch = {
                "trial_order": order,
                "endpoint_error_m": endpoint_error,
                "recorded_success": success,
                "required_success": expected_success,
            }
            inconsistent_success_rows.append(mismatch)
            errors.append(
                f"trial_order {order}: success={success} conflicts with "
                f"endpoint_error_m={endpoint_error:.9g} and the frozen <=0.03 m rule")

    reserve_execution = {}
    if reserve_protocol:
        invalid_roles = [
            row.get("trial_order", "") for row in expected
            if row.get("trial_role") not in {"primary", "reserve"}
        ]
        if invalid_roles:
            errors.append(f"frozen schedule has invalid trial_role rows: {invalid_roles}")
        conditions = sorted({row.get("condition", "") for row in expected})
        for condition in conditions:
            primaries = [
                row for row in expected
                if row.get("condition") == condition and row.get("trial_role") == "primary"
            ]
            reserves = [
                row for row in expected
                if row.get("condition") == condition and row.get("trial_role") == "reserve"
            ]
            try:
                reserves.sort(key=lambda row: int(row.get("reserve_rank", "")))
                ranks = [int(row.get("reserve_rank", "")) for row in reserves]
            except ValueError:
                ranks = []
                errors.append(f"{condition}: reserve_rank must be an integer")
            if ranks != list(range(1, len(reserves) + 1)):
                errors.append(f"{condition}: frozen reserve ranks must be contiguous from 1")
            if any(row.get("reserve_rank", "").strip() for row in primaries):
                errors.append(f"{condition}: primary rows must have blank reserve_rank")
            target = len(primaries)
            valid_count = sum(
                1 for row in primaries
                if parsed_aborted.get(row.get("trial_order", "")) == 0
            )
            needed_reserve_orders = []
            for row in reserves:
                if valid_count >= target:
                    break
                order = row.get("trial_order", "")
                needed_reserve_orders.append(order)
                if order not in observed_by_order:
                    break
                if parsed_aborted.get(order) == 0:
                    valid_count += 1
            observed_reserve_orders = [
                row.get("trial_order", "") for row in reserves
                if row.get("trial_order", "") in observed_by_order
            ]
            if observed_reserve_orders != needed_reserve_orders:
                errors.append(
                    f"{condition}: executed reserves {observed_reserve_orders} must equal "
                    f"the required frozen prefix {needed_reserve_orders}")
            if valid_count < target:
                errors.append(
                    f"{condition}: completed log has {valid_count}/{target} valid trials; "
                    "continue only with the next preregistered reserve")
            reserve_execution[condition] = {
                "required_valid_trials": target,
                "valid_trials": valid_count,
                "primary_trials": len(primaries),
                "primary_aborts": sum(
                    parsed_aborted.get(row.get("trial_order", "")) == 1
                    for row in primaries
                ),
                "reserves_preregistered": len(reserves),
                "reserves_executed": len(observed_reserve_orders),
                "reserve_aborts": sum(
                    parsed_aborted.get(order) == 1 for order in observed_reserve_orders
                ),
                "executed_reserve_trial_orders": observed_reserve_orders,
            }

    return {
        "status": "PASS" if not errors else "FAIL",
        "frozen_schedule": str(schedule),
        "completed_log": str(completed),
        "frozen_schedule_sha256": digest(schedule),
        "completed_log_sha256": digest(completed),
        "frozen_rows": len(expected), "completed_rows": len(observed),
        "identity_fields": list(identity_fields),
        "reserve_protocol": reserve_protocol,
        "reserve_execution_by_condition": reserve_execution,
        "success_threshold_m": SUCCESS_THRESHOLD_M,
        "maximum_allowed_lock_error_rad": MAXIMUM_LOCK_ERROR_RAD,
        "inconsistent_success_rows": inconsistent_success_rows,
        "lock_error_violations": lock_error_violations,
        "changed_identity_rows": changed,
        "errors": errors,
        "claim_boundary": (
            "PASS establishes frozen identity, deterministic reserve use, and "
            "success/lock-threshold integrity; complete raw-file validity still "
            "requires the Push analyzer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("completed", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/real_robot/schedule-completion-audit.json"))
    args = parser.parse_args()
    payload = audit(args.schedule, args.completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
