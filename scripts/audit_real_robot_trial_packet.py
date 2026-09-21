"""Audit one real-robot Push trial evidence packet without hardware access.

The auditor treats a normally completed acquisition and a retained abort as
different outcomes.  Only a complete ``ACQUISITION_COMPLETE_UNASSESSED`` packet
can set ``valid_trial`` to true.  An honestly recorded abort may have intact
packet provenance, but is reported as ``RETAINED_ABORT`` and is never a valid
trial.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    VALID_CONDITIONS, locked_indices_for_condition,
)

NORMAL_STATUS = "ACQUISITION_COMPLETE_UNASSESSED"
ABORT_STATUS_PREFIX = "ABORTED_"
JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

LOG_FILENAMES = {
    "commands": "commands.csv",
    "servo_telemetry": "servo_telemetry.csv",
    "frame_timestamps": "frame_timestamps.csv",
}
MANIFEST_ARTIFACT_KEYS = {
    "commands": "commands",
    "servo_telemetry": "servo_telemetry",
    "frame_timestamps": "frame_timestamps",
    "daheng_video": "daheng_video",
    "directshow_video": "directshow_video",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _artifact_role(path: Path) -> str | None:
    name = path.name.lower()
    if name == "run_manifest.json":
        return "run_manifest"
    for role, filename in LOG_FILENAMES.items():
        if name == filename:
            return role
    if name.startswith("daheng") and name.endswith("_raw.avi"):
        return "daheng_video"
    if name.startswith("directshow") and name.endswith("_raw.avi"):
        return "directshow_video"
    return None


def _discover_artifact_paths(trial_dir: Path) -> list[tuple[str, Path]]:
    if not trial_dir.is_dir():
        return []
    found = []
    for path in trial_dir.iterdir():
        if not path.is_file():
            continue
        role = _artifact_role(path)
        if role is not None:
            found.append((role, path))
    order = {
        "run_manifest": 0,
        "daheng_video": 1,
        "directshow_video": 2,
        "commands": 3,
        "servo_telemetry": 4,
        "frame_timestamps": 5,
    }
    return sorted(found, key=lambda item: (order[item[0]], item[1].name.lower()))


def _build_artifact_table(
    trial_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[Path]], list[str]]:
    table: list[dict[str, Any]] = []
    paths_by_role: dict[str, list[Path]] = {}
    errors: list[str] = []
    for role, path in _discover_artifact_paths(trial_dir):
        paths_by_role.setdefault(role, []).append(path)
        try:
            table.append({
                "role": role,
                "name": path.name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        except OSError as error:
            errors.append(f"cannot hash existing artifact {path.name}: {error}")
    return table, paths_by_role, errors


def _artifact_row(
    artifact_table: list[dict[str, Any]], path: Path,
) -> dict[str, Any] | None:
    resolved = str(path.resolve())
    return next((row for row in artifact_table if row["path"] == resolved), None)


def _one_path(
    paths_by_role: dict[str, list[Path]], role: str, errors: list[str],
) -> Path | None:
    paths = paths_by_role.get(role, [])
    if not paths:
        errors.append(f"normal trial is missing required artifact role {role}")
        return None
    if len(paths) != 1:
        errors.append(
            f"normal trial requires exactly one {role} artifact; found "
            f"{[path.name for path in paths]}"
        )
        return None
    return paths[0]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str | None]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if None in fields or len(set(fields)) != len(fields):
        raise ValueError("CSV header contains a blank or duplicate column")
    if any(None in row for row in rows):
        raise ValueError("CSV row contains more values than its header")
    return fields, rows


def _require_joint_log(
    *,
    path: Path,
    artifact_table: list[dict[str, Any]],
    role: str,
    required_joint_fields: tuple[str, ...],
    manifest_count: object,
    errors: list[str],
) -> None:
    try:
        fields, rows = _read_csv(path)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        errors.append(f"cannot read {role} CSV {path.name}: {error}")
        return
    row = _artifact_row(artifact_table, path)
    if row is not None:
        row.update({"row_count": len(rows), "columns": fields})
    missing = sorted(set(required_joint_fields) - set(fields))
    if missing:
        errors.append(f"{role} is missing five-joint fields: {missing}")
    if not rows:
        errors.append(f"{role} must contain at least one data row")
    elif not missing:
        blank_rows = [
            index for index, item in enumerate(rows, start=2)
            if any(not str(item.get(field) or "").strip() for field in required_joint_fields)
        ]
        if blank_rows:
            errors.append(
                f"{role} has blank five-joint values on CSV lines {blank_rows[:10]}"
            )
    if not _is_integer(manifest_count) or int(manifest_count) < 0:
        errors.append(f"manifest {role}_rows must be a non-negative integer")
    elif int(manifest_count) != len(rows):
        errors.append(
            f"{role} row count mismatch: manifest={manifest_count}, actual={len(rows)}"
        )


def _decode_video(path: Path) -> dict[str, int | float]:
    """Decode every readable frame and return container metadata."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("video cannot be opened by OpenCV")
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError(f"decoded frame {decoded_frames} is empty")
            decoded_frames += 1
    finally:
        capture.release()
    if decoded_frames <= 0:
        raise ValueError("video contains no decodable frame")
    if reported_frames > 0 and reported_frames != decoded_frames:
        raise ValueError(
            f"container reports {reported_frames} frames but {decoded_frames} decode"
        )
    return {
        "decoded_frame_count": decoded_frames,
        "container_reported_frame_count": reported_frames,
        "declared_fps": reported_fps,
        "width": width,
        "height": height,
    }


def _manifest_frame_count(
    manifest: dict[str, Any], camera: str, errors: list[str],
) -> int | None:
    counts = manifest.get("camera_frame_counts")
    if not isinstance(counts, dict):
        errors.append("manifest camera_frame_counts must be an object")
        return None
    value = counts.get(camera)
    if not _is_integer(value) or int(value) <= 0:
        errors.append(
            f"manifest camera_frame_counts.{camera} must be a positive integer"
        )
        return None
    return int(value)


def _validate_video(
    *,
    path: Path,
    camera: str,
    expected_frames: int | None,
    artifact_table: list[dict[str, Any]],
    errors: list[str],
) -> int | None:
    try:
        inspection = _decode_video(path)
    except Exception as error:
        errors.append(f"{camera} video {path.name} is not fully decodable: {error}")
        return None
    row = _artifact_row(artifact_table, path)
    if row is not None:
        row.update(inspection)
    decoded = int(inspection["decoded_frame_count"])
    if expected_frames is not None and decoded != expected_frames:
        errors.append(
            f"{camera} video frame count mismatch: manifest={expected_frames}, "
            f"decoded={decoded}"
        )
    return decoded


def _camera_identity(
    manifest: dict[str, Any], errors: list[str],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "daheng_timestamp": None,
        "directshow_timestamp": None,
        "daheng_video": None,
        "directshow_video": None,
    }
    devices = manifest.get("camera_devices")
    if not isinstance(devices, dict):
        errors.append("manifest camera_devices must be an object")
        return result
    serial = devices.get("daheng_serial")
    index = devices.get("directshow_index")
    if not _is_nonempty_string(serial):
        errors.append("manifest camera_devices.daheng_serial must be non-empty")
    else:
        serial = str(serial).strip()
        result["daheng_timestamp"] = f"daheng_sn_{serial}"
        result["daheng_video"] = f"daheng_{serial}_raw.avi"
    if not _is_integer(index) or int(index) < 0:
        errors.append(
            "manifest camera_devices.directshow_index must be a non-negative integer"
        )
    else:
        result["directshow_timestamp"] = f"directshow_index_{int(index)}"
        result["directshow_video"] = f"directshow_index{int(index)}_raw.avi"
    return result


def _validate_timestamps(
    *,
    path: Path,
    expected_cameras: tuple[str | None, str | None],
    expected_frame_counts: tuple[int | None, int | None],
    artifact_table: list[dict[str, Any]],
    errors: list[str],
) -> None:
    try:
        fields, rows = _read_csv(path)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        errors.append(f"cannot read frame_timestamps CSV {path.name}: {error}")
        return
    artifact = _artifact_row(artifact_table, path)
    if artifact is not None:
        artifact.update({"row_count": len(rows), "columns": fields})
    required = {"camera", "frame_index"}
    if missing := sorted(required - set(fields)):
        errors.append(f"frame_timestamps is missing required fields: {missing}")
        return
    if not rows:
        errors.append("frame_timestamps must contain at least one data row")
        return
    camera_counts = Counter(str(row.get("camera") or "").strip() for row in rows)
    camera_counts.pop("", None)
    if artifact is not None:
        artifact["camera_row_counts"] = dict(sorted(camera_counts.items()))
    for camera, expected_count in zip(expected_cameras, expected_frame_counts):
        if camera is None:
            continue
        observed = camera_counts.get(camera, 0)
        if observed <= 0:
            errors.append(f"frame_timestamps has no rows for manifest camera {camera}")
        if expected_count is not None and observed != expected_count:
            errors.append(
                f"frame_timestamps count mismatch for {camera}: "
                f"manifest={expected_count}, actual={observed}"
            )
    known = {camera for camera in expected_cameras if camera is not None}
    if known:
        unexpected = sorted(set(camera_counts) - known)
        if unexpected:
            errors.append(f"frame_timestamps contains unexpected cameras: {unexpected}")


def _validate_manifest_identity(
    *,
    manifest: dict[str, Any],
    trial_dir: Path,
    expected_trial_id: str | None,
    expected_condition: str | None,
    expected_trajectory_id: str | None,
) -> list[str]:
    errors: list[str] = []
    trial_id = manifest.get("trial_id")
    condition = manifest.get("condition")
    trajectory_id = manifest.get("trajectory_id")
    if not _is_nonempty_string(trial_id):
        errors.append("manifest trial_id must be a non-empty string")
    else:
        trial_id = str(trial_id).strip()
        if trial_id != trial_dir.name:
            errors.append(
                f"manifest trial_id {trial_id!r} does not match directory "
                f"{trial_dir.name!r}"
            )
        if expected_trial_id is not None and trial_id != expected_trial_id:
            errors.append(
                f"manifest trial_id {trial_id!r} does not match expected "
                f"{expected_trial_id!r}"
            )
    if condition not in VALID_CONDITIONS:
        errors.append(f"manifest condition must be one of {VALID_CONDITIONS}")
    elif expected_condition is not None and condition != expected_condition:
        errors.append(
            f"manifest condition {condition!r} does not match expected "
            f"{expected_condition!r}"
        )
    if not _is_nonempty_string(trajectory_id):
        errors.append("manifest trajectory_id must be a non-empty string")
    elif (expected_trajectory_id is not None
          and str(trajectory_id).strip() != expected_trajectory_id):
        errors.append(
            f"manifest trajectory_id {trajectory_id!r} does not match expected "
            f"{expected_trajectory_id!r}"
        )
    waypoint_digest = manifest.get("waypoint_sha256")
    if waypoint_digest is not None and (
        not isinstance(waypoint_digest, str)
        or SHA256_PATTERN.fullmatch(waypoint_digest) is None
    ):
        errors.append("manifest waypoint_sha256 must be a 64-digit hexadecimal digest")
    if condition in VALID_CONDITIONS:
        expected_indices = locked_indices_for_condition(str(condition))
        expected_locked_names = [JOINT_NAMES[index] for index in expected_indices]
        expected_locked = expected_locked_names[0] if len(expected_locked_names) == 1 else None
        locked = manifest.get("locked_joint")
        locked_target = manifest.get("locked_target_raw")
        if locked != expected_locked:
            errors.append(
                f"manifest locked_joint must be {expected_locked!r} for {condition}"
            )
        if not expected_indices:
            if locked_target is not None:
                errors.append("intact manifest locked_target_raw must be null")
        elif len(expected_indices) == 1 and not _is_integer(locked_target):
            errors.append(
                f"{condition} manifest locked_target_raw must be an integer"
            )
        locked_names = manifest.get("locked_joints")
        locked_targets = manifest.get("locked_targets_raw")
        if locked_names is not None and locked_names != expected_locked_names:
            errors.append(f"manifest locked_joints must be {expected_locked_names!r}")
        if len(expected_indices) > 1:
            if locked_names != expected_locked_names:
                errors.append("multi-lock manifest must declare every locked joint")
            if not isinstance(locked_targets, dict) or set(locked_targets) != set(expected_locked_names):
                errors.append("multi-lock manifest locked_targets_raw must cover every locked joint")
            elif any(not _is_integer(value) for value in locked_targets.values()):
                errors.append("multi-lock manifest locked_targets_raw values must be integers")
    return errors


def _validate_manifest_artifact_names(
    manifest: dict[str, Any], paths_by_role: dict[str, list[Path]], errors: list[str],
) -> None:
    declared = manifest.get("artifacts")
    if declared is None:
        errors.append("normal manifest is missing the artifacts filename map")
        return
    if not isinstance(declared, dict):
        errors.append("manifest artifacts must be an object when present")
        return
    for manifest_key, role in MANIFEST_ARTIFACT_KEYS.items():
        value = declared.get(manifest_key)
        if not _is_nonempty_string(value):
            errors.append(f"manifest artifacts.{manifest_key} must name a file")
            continue
        declared_path = Path(str(value))
        if declared_path.name != str(value):
            errors.append(
                f"manifest artifacts.{manifest_key} must be a trial-local filename"
            )
            continue
        actual = [path.name for path in paths_by_role.get(role, [])]
        if str(value) not in actual:
            errors.append(
                f"manifest artifacts.{manifest_key}={value!r} is not the actual "
                f"{role} artifact {actual}"
            )


def audit_trial_packet(
    trial_dir: Path,
    *,
    expected_trial_id: str | None = None,
    expected_condition: str | None = None,
    expected_trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Return the machine-readable integrity and validity audit for one trial."""
    trial_dir = trial_dir.resolve()
    artifact_table, paths_by_role, artifact_errors = _build_artifact_table(trial_dir)
    errors = list(artifact_errors)
    manifest_path = trial_dir / "run_manifest.json"
    manifest: dict[str, Any] | None = None
    if not trial_dir.is_dir():
        errors.append(f"trial directory does not exist: {trial_dir}")
    if not manifest_path.is_file():
        errors.append("trial packet is missing run_manifest.json")
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(loaded, dict):
                raise ValueError("top-level JSON value is not an object")
            manifest = loaded
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"cannot read run_manifest.json: {error}")

    identity: dict[str, Any] = {}
    manifest_status: object = None
    is_abort = False
    is_normal = False
    if manifest is not None:
        manifest_status = manifest.get("status")
        identity = {
            name: manifest.get(name)
            for name in (
                "trial_id", "trajectory_id", "condition", "locked_joint",
                "locked_target_raw", "status",
            )
        }
        errors.extend(_validate_manifest_identity(
            manifest=manifest,
            trial_dir=trial_dir,
            expected_trial_id=expected_trial_id,
            expected_condition=expected_condition,
            expected_trajectory_id=expected_trajectory_id,
        ))
        is_normal = manifest_status == NORMAL_STATUS
        is_abort = (
            isinstance(manifest_status, str)
            and manifest_status.startswith(ABORT_STATUS_PREFIX)
        )
        if not is_normal and not is_abort:
            errors.append(
                f"manifest status {manifest_status!r} is neither normal completion "
                "nor a retained abort"
            )
        if is_abort:
            for name in ("aborted_utc", "failure_type", "failure_message"):
                if not _is_nonempty_string(manifest.get(name)):
                    errors.append(f"aborted manifest requires non-empty {name}")

    normal_validation_errors: list[str] = []
    if manifest is not None and is_normal:
        _validate_manifest_artifact_names(
            manifest, paths_by_role, normal_validation_errors
        )
        required_paths = {
            role: _one_path(paths_by_role, role, normal_validation_errors)
            for role in (
                "daheng_video", "directshow_video", "commands",
                "servo_telemetry", "frame_timestamps",
            )
        }
        daheng_frames = _manifest_frame_count(
            manifest, "daheng", normal_validation_errors
        )
        directshow_frames = _manifest_frame_count(
            manifest, "directshow", normal_validation_errors
        )
        for camera, role, expected in (
            ("daheng", "daheng_video", daheng_frames),
            ("directshow", "directshow_video", directshow_frames),
        ):
            path = required_paths[role]
            if path is not None:
                _validate_video(
                    path=path,
                    camera=camera,
                    expected_frames=expected,
                    artifact_table=artifact_table,
                    errors=normal_validation_errors,
                )
        commands = required_paths["commands"]
        if commands is not None:
            _require_joint_log(
                path=commands,
                artifact_table=artifact_table,
                role="commands",
                required_joint_fields=tuple(
                    f"{joint}_target_raw" for joint in JOINT_NAMES
                ),
                manifest_count=manifest.get("command_rows"),
                errors=normal_validation_errors,
            )
        telemetry = required_paths["servo_telemetry"]
        if telemetry is not None:
            _require_joint_log(
                path=telemetry,
                artifact_table=artifact_table,
                role="servo_telemetry",
                required_joint_fields=tuple(
                    field
                    for joint in JOINT_NAMES
                    for field in (f"{joint}_position_raw", f"{joint}_target_raw")
                ),
                manifest_count=manifest.get("telemetry_rows"),
                errors=normal_validation_errors,
            )
        camera_identity = _camera_identity(manifest, normal_validation_errors)
        for role in ("daheng_video", "directshow_video"):
            path = required_paths[role]
            expected_name = camera_identity[role]
            if (path is not None and expected_name is not None
                    and path.name != expected_name):
                normal_validation_errors.append(
                    f"{role} filename {path.name!r} does not match manifest camera "
                    f"identity {expected_name!r}"
                )
        timestamps = required_paths["frame_timestamps"]
        if timestamps is not None:
            _validate_timestamps(
                path=timestamps,
                expected_cameras=(
                    camera_identity["daheng_timestamp"],
                    camera_identity["directshow_timestamp"],
                ),
                expected_frame_counts=(daheng_frames, directshow_frames),
                artifact_table=artifact_table,
                errors=normal_validation_errors,
            )
        errors.extend(normal_validation_errors)

    packet_integrity_status = "PASS" if not errors else "FAIL"
    valid_trial = bool(is_normal and packet_integrity_status == "PASS")
    if valid_trial:
        status = "PASS"
        trial_validity_status = "VALID_UNASSESSED"
    elif is_abort and packet_integrity_status == "PASS":
        status = "RETAINED_ABORT"
        trial_validity_status = "INVALID_ABORTED"
    else:
        status = "FAIL"
        trial_validity_status = "INVALID_PACKET"
    return {
        "status": status,
        "packet_integrity_status": packet_integrity_status,
        "trial_validity_status": trial_validity_status,
        "valid_trial": valid_trial,
        "trial_directory": str(trial_dir),
        "manifest_status": manifest_status,
        "identity": identity,
        "artifact_count": len(artifact_table),
        "artifacts": artifact_table,
        "errors": errors,
        "claim_boundary": (
            "PASS establishes only raw packet identity, readability, hashes, "
            "five-joint log presence, and frame-count consistency. It does not "
            "establish reach, contact, task success, or a learned-method result. "
            "RETAINED_ABORT is preserved evidence and is never a valid trial."
        ),
    }


# Short alias matching the naming convention of other audit scripts.
audit = audit_trial_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="optional JSON file; the full audit is always emitted on stdout",
    )
    parser.add_argument("--expected-trial-id")
    parser.add_argument("--expected-condition", choices=VALID_CONDITIONS)
    parser.add_argument("--expected-trajectory-id")
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_trial_packet(
        args.trial_dir,
        expected_trial_id=args.expected_trial_id,
        expected_condition=args.expected_condition,
        expected_trajectory_id=args.expected_trajectory_id,
    )
    if args.output is not None:
        _atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] == "PASS":
        return 0
    if payload["status"] == "RETAINED_ABORT":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
