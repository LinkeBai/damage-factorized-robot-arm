"""Hard preflight gate for the frozen original-5DoF real Push experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml


EXPECTED_SHA256 = "79139bca3b61866643e00ef35d724cdd4185fb14a8f115faa942635f27f4510d"
EXPECTED_CONDITIONS = {"intact": 5, "D2": 10, "D3": 10}
EXPECTED_METHODS = {"nominal", "global_matched"}
LEVEL_A_CONDITIONS = ("intact", "D2", "D3")
LEVEL_A_POSITION_ID = "A"
LEVEL_A_PRIMARY_TRIALS_PER_CONDITION = 10
LEVEL_A_RESERVES_PER_CONDITION = 10
MAXIMUM_LOCK_ERROR_RAD = math.radians(3.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(value) -> bool:
    return value is not None and str(value).strip() != ""


def level_a_readiness_checks(manifest: dict, errors: list[str]) -> dict:
    """Validate frozen PASS artifacts without probing cameras or hardware."""
    result = {
        "servo_readiness": "NOT_CHECKED",
        "camera_synchronization": "NOT_CHECKED",
    }
    hardware = manifest.get("hardware", {})
    cameras = manifest.get("cameras", {})
    servo_value = hardware.get("servo_readiness_audit_file")
    camera_value = cameras.get("synchronization_audit_file")

    if nonempty(servo_value) and Path(str(servo_value)).is_file():
        servo_path = Path(str(servo_value))
        try:
            payload = json.loads(servo_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            errors.append("servo readiness audit is not valid JSON")
        else:
            if payload.get("status") != "PASS":
                errors.append("servo readiness audit status is not PASS")
            if payload.get("read_only") is not True:
                errors.append("servo readiness audit must record read_only=true")
            if not nonempty(payload.get("timestamp_utc")):
                errors.append("servo readiness audit must record timestamp_utc")
            servos = payload.get("servos")
            if not isinstance(servos, list):
                errors.append("servo readiness audit must contain a servos list")
            else:
                listed_ids = [
                    item.get("servo_id") for item in servos if isinstance(item, dict)
                ]
                responded_ids = [
                    item.get("servo_id") for item in servos
                    if isinstance(item, dict) and item.get("responded") is True
                ]
                if (len(listed_ids) != 5
                        or not all(isinstance(value, int) for value in listed_ids)
                        or set(listed_ids) != {1, 2, 3, 4, 5}
                        or len(responded_ids) != 5
                        or not all(isinstance(value, int) for value in responded_ids)
                        or set(responded_ids) != {1, 2, 3, 4, 5}):
                    errors.append(
                        "servo readiness audit must list exactly IDs 1-5 once with "
                        "responded=true")
            recorded_hash = hardware.get("servo_readiness_audit_sha256")
            if recorded_hash != sha256(servo_path):
                errors.append("servo readiness audit hash does not match the frozen manifest")
            result["servo_readiness"] = payload.get("status", "INVALID")

    if nonempty(camera_value) and Path(str(camera_value)).is_file():
        camera_path = Path(str(camera_value))
        try:
            payload = json.loads(camera_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            errors.append("camera synchronization audit is not valid JSON")
        else:
            if payload.get("status") != "PASS":
                errors.append("camera synchronization audit status is not PASS")
            if not nonempty(payload.get("timestamp_utc")):
                errors.append("camera synchronization audit must record timestamp_utc")
            if payload.get("left_camera_serial") != cameras.get("left_eye_to_hand_serial"):
                errors.append("camera synchronization audit left-camera serial mismatch")
            if payload.get("horizontal_camera_serial") != cameras.get(
                    "horizontal_eye_to_hand_serial"):
                errors.append("camera synchronization audit horizontal-camera serial mismatch")
            try:
                observed_error = float(payload["maximum_observed_sync_error_ms"])
                allowed_error = float(cameras["maximum_allowed_sync_error_ms"])
                if (not math.isfinite(observed_error) or observed_error < 0
                        or observed_error > allowed_error):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(
                    "camera synchronization audit must record a finite non-negative "
                    "maximum_observed_sync_error_ms within the frozen limit")
            recorded_hash = cameras.get("synchronization_audit_sha256")
            if recorded_hash != sha256(camera_path):
                errors.append(
                    "camera synchronization audit hash does not match the frozen manifest")
            result["camera_synchronization"] = payload.get("status", "INVALID")
    return result


def audit(manifest_path: Path, schedule_path: Path, require_paths: bool = True,
          mode: str = "level_b") -> dict:
    if mode not in {"level_a", "level_b"}:
        raise ValueError(f"unknown preflight mode: {mode}")
    errors: list[str] = []
    warnings: list[str] = []
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schedule_hash = sha256(schedule_path)
    if mode == "level_b" and schedule_hash != EXPECTED_SHA256:
        errors.append(f"schedule SHA-256 mismatch: {schedule_hash}")

    with schedule_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        schedule_fields = set(reader.fieldnames or [])
        rows = list(reader)
    base_schedule_fields = {
        "trial_order", "pair_id", "condition", "method", "position_id",
    }
    if missing := base_schedule_fields - schedule_fields:
        errors.append(f"schedule missing required fields: {sorted(missing)}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("condition", ""), row.get("pair_id", ""))].append(row)
    condition_pairs = Counter(condition for condition, _ in groups)
    level_a_schedule = {}
    if mode == "level_b":
        if len(rows) != 50:
            errors.append(f"schedule must contain 50 trials, found {len(rows)}")
        if dict(condition_pairs) != EXPECTED_CONDITIONS:
            errors.append(
                f"condition pair counts must be {EXPECTED_CONDITIONS}, found {dict(condition_pairs)}")
        for key, pair in groups.items():
            methods = {row["method"] for row in pair}
            positions = {row["position_id"] for row in pair}
            if methods != EXPECTED_METHODS:
                errors.append(f"pair {key} methods must be {sorted(EXPECTED_METHODS)}, found {sorted(methods)}")
            if len(positions) != 1:
                errors.append(f"pair {key} has mismatched physical reset positions")
    else:
        reserve_fields = {"trajectory_id", "trial_role", "reserve_rank"}
        if missing := reserve_fields - schedule_fields:
            errors.append(
                f"Level-A schedule missing preregistered reserve fields: {sorted(missing)}")
        primary_rows = [row for row in rows if row.get("trial_role") == "primary"]
        reserve_rows = [row for row in rows if row.get("trial_role") == "reserve"]
        invalid_roles = [
            row.get("trial_order", "") for row in rows
            if row.get("trial_role") not in {"primary", "reserve"}
        ]
        if invalid_roles:
            errors.append(f"Level-A schedule has invalid trial_role rows: {invalid_roles}")
        primary_counts = Counter(row.get("condition", "") for row in primary_rows)
        reserve_counts = Counter(row.get("condition", "") for row in reserve_rows)
        expected_primary = {
            condition: LEVEL_A_PRIMARY_TRIALS_PER_CONDITION
            for condition in LEVEL_A_CONDITIONS
        }
        if dict(primary_counts) != expected_primary:
            errors.append(
                f"Level-A primary counts must be {expected_primary}, "
                f"found {dict(primary_counts)}")
        expected_reserves = {
            condition: LEVEL_A_RESERVES_PER_CONDITION
            for condition in LEVEL_A_CONDITIONS
        }
        if dict(reserve_counts) != expected_reserves:
            errors.append(
                f"Level-A reserve counts must be {expected_reserves}, "
                f"found {dict(reserve_counts)}")
        if {row.get("method") for row in rows} != {"fixed_safe_trajectory"}:
            errors.append("Level-A method must be fixed_safe_trajectory for every trial")
        positions = {row.get("position_id") for row in rows}
        if positions != {LEVEL_A_POSITION_ID}:
            errors.append(
                f"Level-A every primary/reserve trial must use frozen position A; "
                f"found {sorted(str(value) for value in positions)}")
        if any(not nonempty(row.get("trajectory_id")) for row in rows):
            errors.append("Level-A requires a validated trajectory_id on every trial")
        if any(nonempty(row.get("reserve_rank")) for row in primary_rows):
            errors.append("Level-A primary rows must have blank reserve_rank")
        for condition in LEVEL_A_CONDITIONS:
            selected = [row for row in reserve_rows if row.get("condition") == condition]
            try:
                ranks = sorted(int(row.get("reserve_rank", "")) for row in selected)
            except ValueError:
                ranks = []
                errors.append(f"Level-A {condition} reserve_rank must be an integer")
            if ranks != list(range(1, len(selected) + 1)):
                errors.append(
                    f"Level-A {condition} reserve ranks must be unique and contiguous from 1")
        try:
            primary_orders = [int(row.get("trial_order", "")) for row in primary_rows]
            reserve_orders = [int(row.get("trial_order", "")) for row in reserve_rows]
            if (primary_orders and reserve_orders
                    and max(primary_orders) >= min(reserve_orders)):
                errors.append(
                    "Level-A reserve block must follow every frozen primary trial")
        except ValueError:
            pass
        level_a_schedule = {
            "frozen_position_id": LEVEL_A_POSITION_ID,
            "primary_trials_by_condition": dict(primary_counts),
            "reserve_trials_by_condition": dict(reserve_counts),
            "reserve_policy": (
                "all primary rows are mandatory; consume only the frozen "
                "per-condition reserve prefix needed to reach 10 valid trials"
            ),
        }
    orders = [row.get("trial_order", "") for row in rows]
    if len(set(orders)) != len(orders) or any(not value for value in orders):
        errors.append("trial_order must be populated and unique")
    try:
        numeric_orders = sorted(int(value) for value in orders)
        if numeric_orders != list(range(1, len(rows) + 1)):
            errors.append("trial_order must be contiguous from 1")
    except ValueError:
        errors.append("trial_order must contain integers")

    required_scalar_paths = {
        "session_id": manifest.get("session_id"),
        "date_local": manifest.get("date_local"),
        "operator": manifest.get("operator"),
        "hardware.robot_asset_id": manifest.get("hardware", {}).get("robot_asset_id"),
        "hardware.gripper_asset_id": manifest.get("hardware", {}).get("gripper_asset_id"),
        "hardware.block_asset_id": manifest.get("hardware", {}).get("block_asset_id"),
        "cameras.left_eye_to_hand_serial": manifest.get("cameras", {}).get("left_eye_to_hand_serial"),
        "cameras.horizontal_eye_to_hand_serial": manifest.get("cameras", {}).get("horizontal_eye_to_hand_serial"),
        "cameras.left_calibration_file": manifest.get("cameras", {}).get("left_calibration_file"),
        "cameras.horizontal_calibration_file": manifest.get("cameras", {}).get("horizontal_calibration_file"),
        "cameras.synchronization_event_video": manifest.get("cameras", {}).get("synchronization_event_video"),
        "safety.maximum_commanded_joint_speed_rad_s": manifest.get("safety", {}).get("maximum_commanded_joint_speed_rad_s"),
        "safety.maximum_allowed_lock_error_rad": manifest.get("safety", {}).get("maximum_allowed_lock_error_rad"),
        "safety.workspace_boundary_description": manifest.get("safety", {}).get("workspace_boundary_description"),
        "randomization.schedule_file": manifest.get("randomization", {}).get("schedule_file"),
        "randomization.schedule_sha256_before_trials": manifest.get("randomization", {}).get("schedule_sha256_before_trials"),
        "randomization.physical_reset_fixture_description": manifest.get("randomization", {}).get("physical_reset_fixture_description"),
        "randomization.action_library_hash": manifest.get("randomization", {}).get("action_library_hash"),
        "randomization.action_library_file": manifest.get("randomization", {}).get("action_library_file"),
        "randomization.action_library_validation_file": manifest.get("randomization", {}).get("action_library_validation_file"),
        "freeze_record.freeze_timestamp_local": manifest.get("freeze_record", {}).get("freeze_timestamp_local"),
        "freeze_record.operator_signature_or_initials": manifest.get("freeze_record", {}).get("operator_signature_or_initials"),
    }
    if mode == "level_a":
        required_scalar_paths.update({
            "hardware.servo_readiness_audit_file": manifest.get(
                "hardware", {}).get("servo_readiness_audit_file"),
            "hardware.servo_readiness_audit_sha256": manifest.get(
                "hardware", {}).get("servo_readiness_audit_sha256"),
            "cameras.synchronization_audit_file": manifest.get(
                "cameras", {}).get("synchronization_audit_file"),
            "cameras.synchronization_audit_sha256": manifest.get(
                "cameras", {}).get("synchronization_audit_sha256"),
        })
    if mode == "level_b":
        required_scalar_paths.update({
            "randomization.action_interface_bridge_file": manifest.get("randomization", {}).get("action_interface_bridge_file"),
            "randomization.action_interface_validation_file": manifest.get("randomization", {}).get("action_interface_validation_file"),
        })
    for name, value in required_scalar_paths.items():
        if not nonempty(value):
            errors.append(f"manifest field is blank: {name}")
    if mode == "level_a":
        try:
            configured_lock_error = float(
                manifest.get("safety", {}).get("maximum_allowed_lock_error_rad"))
            if (not math.isfinite(configured_lock_error)
                    or configured_lock_error < 0
                    or configured_lock_error > MAXIMUM_LOCK_ERROR_RAD + 1e-12):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                "safety.maximum_allowed_lock_error_rad must not exceed "
                f"3.5 deg ({MAXIMUM_LOCK_ERROR_RAD:.9g} rad)")

    required_true = {
        "hardware.emergency_stop_checked": manifest.get("hardware", {}).get("emergency_stop_checked"),
        "hardware.joint_direction_check_complete": manifest.get("hardware", {}).get("joint_direction_check_complete"),
        "hardware.low_speed_stop_check_complete": manifest.get("hardware", {}).get("low_speed_stop_check_complete"),
        "freeze_record.frozen_before_first_method_trial": manifest.get("freeze_record", {}).get("frozen_before_first_method_trial"),
    }
    if mode == "level_b":
        required_true["randomization.learned_method_comparison_authorized"] = (
            manifest.get("randomization", {}).get("learned_method_comparison_authorized"))
    for name, value in required_true.items():
        if value is not True:
            errors.append(f"manifest field must be true: {name}")

    recorded_hash = manifest.get("randomization", {}).get("schedule_sha256_before_trials")
    if nonempty(recorded_hash) and recorded_hash != schedule_hash:
        errors.append("manifest schedule hash does not match the schedule file")
    library_value = manifest.get("randomization", {}).get("action_library_file")
    library_hash = manifest.get("randomization", {}).get("action_library_hash")
    validation_value = manifest.get("randomization", {}).get("action_library_validation_file")
    if nonempty(library_value) and Path(str(library_value)).is_file():
        actual_library_hash = sha256(Path(str(library_value)))
        if library_hash != actual_library_hash:
            errors.append("manifest action-library hash does not match the library file")
    if nonempty(validation_value) and Path(str(validation_value)).is_file():
        try:
            validation = json.loads(Path(str(validation_value)).read_text(encoding="utf-8"))
            if validation.get("status") != "PASS":
                errors.append("action-library validation status is not PASS")
            if validation.get("library_sha256") != library_hash:
                errors.append("action-library validation hash does not match manifest")
        except (json.JSONDecodeError, OSError):
            errors.append("action-library validation file is not valid JSON")

    task = manifest.get("frozen_task_definition", {})
    frozen_invariants = {
        "endpoint_error_success_threshold_m": 0.03,
        "near_contact_threshold_m": 0.01,
        "preserve_aborts_and_failures": True,
    }
    if mode == "level_a":
        frozen_invariants.update({
            "frozen_position_id": "A",
            "minimum_valid_trials_per_condition": 10,
            "reserve_trials_per_condition": 10,
            "reserve_execution_rule": "frozen_prefix_until_10_valid",
        })
    if mode == "level_b":
        frozen_invariants.update({
            "primary_reference_method": "nominal",
            "primary_candidate_method": "global_matched",
            "minimum_complete_pairs_per_fault": 10,
        })
    for name, expected in frozen_invariants.items():
        if task.get(name) != expected:
            errors.append(f"frozen_task_definition.{name} must equal {expected!r}")

    path_fields = [
        ("cameras.left_calibration_file", manifest.get("cameras", {}).get("left_calibration_file"), "file"),
        ("cameras.horizontal_calibration_file", manifest.get("cameras", {}).get("horizontal_calibration_file"), "file"),
        ("cameras.synchronization_event_video", manifest.get("cameras", {}).get("synchronization_event_video"), "file"),
        ("data_roots.left_video_directory", manifest.get("data_roots", {}).get("left_video_directory"), "dir"),
        ("data_roots.horizontal_video_directory", manifest.get("data_roots", {}).get("horizontal_video_directory"), "dir"),
        ("data_roots.control_log_directory", manifest.get("data_roots", {}).get("control_log_directory"), "dir"),
        ("data_roots.backup_copy_1", manifest.get("data_roots", {}).get("backup_copy_1"), "dir"),
        ("data_roots.backup_copy_2", manifest.get("data_roots", {}).get("backup_copy_2"), "dir"),
        ("randomization.action_library_file", manifest.get("randomization", {}).get("action_library_file"), "file"),
        ("randomization.action_library_validation_file", manifest.get("randomization", {}).get("action_library_validation_file"), "file"),
    ]
    if mode == "level_a":
        path_fields.extend([
            ("hardware.servo_readiness_audit_file", manifest.get(
                "hardware", {}).get("servo_readiness_audit_file"), "file"),
            ("cameras.synchronization_audit_file", manifest.get(
                "cameras", {}).get("synchronization_audit_file"), "file"),
        ])
    if mode == "level_b":
        path_fields.extend([
            ("randomization.action_interface_bridge_file", manifest.get("randomization", {}).get("action_interface_bridge_file"), "file"),
            ("randomization.action_interface_validation_file", manifest.get("randomization", {}).get("action_interface_validation_file"), "file"),
        ])
    if require_paths:
        for name, value, kind in path_fields:
            if not nonempty(value):
                errors.append(f"manifest path is blank: {name}")
                continue
            path = Path(str(value))
            exists = path.is_file() if kind == "file" else path.is_dir()
            if not exists:
                errors.append(f"manifest {kind} does not exist: {name}={value}")
    else:
        warnings.append("filesystem path existence checks disabled")

    readiness_checks = (
        level_a_readiness_checks(manifest, errors) if mode == "level_a" else {}
    )

    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": mode,
        "manifest": str(manifest_path),
        "schedule": str(schedule_path),
        "schedule_sha256": schedule_hash,
        "schedule_trials": len(rows),
        "schedule_pairs": len(groups),
        "condition_pairs": dict(condition_pairs),
        "level_a_schedule": level_a_schedule,
        "readiness_checks": readiness_checks,
        "errors": errors,
        "warnings": warnings,
        "authorization": (
            ("LEVEL_A_TRIALS_MAY_START" if mode == "level_a"
             else "LEVEL_B_METHOD_TRIALS_MAY_START") if not errors
            else "DO_NOT_START_FORMAL_TRIALS"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("level_a", "level_b"), default="level_b")
    parser.add_argument("--schedule", type=Path,
                        default=Path("data/real_robot/push_schedule_seed20260901.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/real_robot/preflight-audit.json"))
    parser.add_argument("--skip-path-existence", action="store_true",
                        help="Schema test only; forbidden for the formal preflight")
    args = parser.parse_args()
    payload = audit(args.manifest, args.schedule, not args.skip_path_existence, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
