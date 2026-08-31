"""Hard preflight gate for the frozen original-5DoF real Push experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


EXPECTED_SHA256 = "79139bca3b61866643e00ef35d724cdd4185fb14a8f115faa942635f27f4510d"
EXPECTED_CONDITIONS = {"intact": 5, "D2": 10, "D3": 10}
EXPECTED_METHODS = {"nominal", "global_matched"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(value) -> bool:
    return value is not None and str(value).strip() != ""


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
        rows = list(csv.DictReader(handle))
    expected_trials = 30 if mode == "level_a" else 50
    if len(rows) != expected_trials:
        errors.append(f"schedule must contain {expected_trials} trials, found {len(rows)}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["pair_id"])].append(row)
    condition_pairs = Counter(condition for condition, _ in groups)
    if mode == "level_b":
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
        condition_trials = Counter(row["condition"] for row in rows)
        expected_level_a = {"intact": 10, "D2": 10, "D3": 10}
        if dict(condition_trials) != expected_level_a:
            errors.append(
                f"Level-A trial counts must be {expected_level_a}, found {dict(condition_trials)}")
        if {row.get("method") for row in rows} != {"fixed_safe_trajectory"}:
            errors.append("Level-A method must be fixed_safe_trajectory for every trial")
        if any(not nonempty(row.get("trajectory_id")) for row in rows):
            errors.append("Level-A requires a validated trajectory_id on every trial")
    orders = [row["trial_order"] for row in rows]
    if len(set(orders)) != len(orders) or any(not value for value in orders):
        errors.append("trial_order must be populated and unique")

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
        "freeze_record.freeze_timestamp_local": manifest.get("freeze_record", {}).get("freeze_timestamp_local"),
        "freeze_record.operator_signature_or_initials": manifest.get("freeze_record", {}).get("operator_signature_or_initials"),
    }
    if mode == "level_b":
        required_scalar_paths.update({
            "randomization.action_interface_bridge_file": manifest.get("randomization", {}).get("action_interface_bridge_file"),
            "randomization.action_interface_validation_file": manifest.get("randomization", {}).get("action_interface_validation_file"),
        })
    for name, value in required_scalar_paths.items():
        if not nonempty(value):
            errors.append(f"manifest field is blank: {name}")

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

    task = manifest.get("frozen_task_definition", {})
    frozen_invariants = {
        "endpoint_error_success_threshold_m": 0.03,
        "near_contact_threshold_m": 0.01,
        "preserve_aborts_and_failures": True,
    }
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
    ]
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

    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": mode,
        "manifest": str(manifest_path),
        "schedule": str(schedule_path),
        "schedule_sha256": schedule_hash,
        "schedule_trials": len(rows),
        "schedule_pairs": len(groups),
        "condition_pairs": dict(condition_pairs),
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
