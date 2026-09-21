"""Record a read-only dual-camera synchronization audit.

Running this command immediately opens only the two cameras.  It never opens,
reads, or writes the servo bus.  The saved AVI frames are the native,
unannotated frames written by the acquisition recorders used for real Push
trials.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_real_push_fixed_trajectory import (  # noqa: E402
    DAHENG_SERIAL,
    DIRECTSHOW_INDEX,
    BackgroundFailure,
    DahengRecorder,
    DirectShowRecorder,
    FrameTimestampLog,
    atomic_write_json,
    sha256_file,
    utc_now,
    validate_camera_settings,
    validate_recorder_cadence,
    validate_video_file,
)


LEFT_CAMERA_SERIAL = DAHENG_SERIAL
HORIZONTAL_CAMERA_SERIAL = "USB VID_32E6 PID_9211"
LEFT_TIMESTAMP_CAMERA = f"daheng_sn_{LEFT_CAMERA_SERIAL}"
HORIZONTAL_TIMESTAMP_CAMERA = f"directshow_index_{DIRECTSHOW_INDEX}"
MAXIMUM_ALLOWED_SYNC_ERROR_MS = 50.0
DEFAULT_DURATION_S = 5.0
DEFAULT_VIDEO_FPS = 20.0
DEFAULT_OUTPUT_DIR = ROOT / "results/real_robot/camera_sync_audit"
DEFAULT_CAMERA_SETTINGS = ROOT / "results/real_robot/camera_settings_selected.json"


def _strictly_increasing(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one capture midpoint")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError(f"{name} capture midpoints must be integer nanoseconds")
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise ValueError(f"{name} capture midpoints must be strictly increasing")
    return result


def nearest_midpoint_pairs(
    left_midpoints_ns: Sequence[int], horizontal_midpoints_ns: Sequence[int],
) -> list[tuple[int, int, float]]:
    """Pair each left frame to the nearest horizontal capture midpoint.

    The returned tuples are ``(left_index, horizontal_index, error_ms)``.  A
    midpoint exactly between two horizontal frames is paired with the earlier
    frame, making the result deterministic.
    """
    left = _strictly_increasing(left_midpoints_ns, "left")
    horizontal = _strictly_increasing(horizontal_midpoints_ns, "horizontal")
    pairs: list[tuple[int, int, float]] = []
    for left_index, midpoint in enumerate(left):
        insertion = bisect.bisect_left(horizontal, midpoint)
        if insertion == 0:
            horizontal_index = 0
        elif insertion == len(horizontal):
            horizontal_index = len(horizontal) - 1
        else:
            earlier_error = midpoint - horizontal[insertion - 1]
            later_error = horizontal[insertion] - midpoint
            horizontal_index = insertion - 1 if earlier_error <= later_error else insertion
        error_ms = abs(midpoint - horizontal[horizontal_index]) / 1_000_000.0
        pairs.append((left_index, horizontal_index, error_ms))
    return pairs


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not math.isfinite(percentile_value) or not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calculate_sync_metrics(
    left_midpoints_ns: Sequence[int], horizontal_midpoints_ns: Sequence[int],
) -> dict[str, float | int]:
    """Calculate nearest-midpoint max and p95 synchronization errors."""
    pairs = nearest_midpoint_pairs(left_midpoints_ns, horizontal_midpoints_ns)
    errors_ms = [pair[2] for pair in pairs]
    return {
        "sample_count": len(pairs),
        "maximum_observed_sync_error_ms": max(errors_ms),
        "p95_sync_error_ms": percentile(errors_ms, 95.0),
    }


def load_capture_midpoints(
    path: Path, *, start_ns: int | None = None, end_ns: int | None = None,
) -> dict[str, list[int]]:
    """Load per-camera capture midpoints from a ``FrameTimestampLog`` CSV."""
    result = {LEFT_TIMESTAMP_CAMERA: [], HORIZONTAL_TIMESTAMP_CAMERA: []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"camera", "capture_mid_monotonic_ns"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("frame timestamp log is missing required columns")
        for row in reader:
            camera = row["camera"]
            if camera not in result:
                raise ValueError(f"unexpected camera in frame timestamp log: {camera!r}")
            try:
                midpoint = int(row["capture_mid_monotonic_ns"])
            except (TypeError, ValueError) as exc:
                raise ValueError("capture midpoint must be an integer") from exc
            if start_ns is not None and midpoint < start_ns:
                continue
            if end_ns is not None and midpoint > end_ns:
                continue
            result[camera].append(midpoint)
    for camera, values in result.items():
        _strictly_increasing(values, camera)
    return result


def _artifact(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _existing_artifacts(paths: dict[str, Path]) -> dict[str, dict[str, str | int]]:
    return {
        name: _artifact(path)
        for name, path in paths.items()
        if path.is_file()
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and positive")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0.0:
        raise ValueError("--video-fps must be finite and positive")
    if (not math.isfinite(args.camera_ready_timeout_s)
            or args.camera_ready_timeout_s <= 0.0):
        raise ValueError("--camera-ready-timeout-s must be finite and positive")
    if not args.camera_settings.is_file():
        raise ValueError(f"camera settings file does not exist: {args.camera_settings}")


def run_audit(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Acquire both cameras and write a PASS/FAIL JSON audit."""
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    audit_path = (args.audit.resolve() if args.audit is not None
                  else output_dir / "camera_sync_audit.json")
    paths = {
        "left_video": output_dir / f"daheng_{LEFT_CAMERA_SERIAL}_raw.avi",
        "horizontal_video": output_dir / "horizontal_usb_vid_32e6_pid_9211_raw.avi",
        "frame_timestamps": output_dir / "frame_timestamps.csv",
    }
    targets = [*paths.values(), audit_path]
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("audit and capture artifact paths must be distinct")
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing audit artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    settings_payload = json.loads(
        args.camera_settings.read_text(encoding="utf-8-sig")
    )
    settings = validate_camera_settings(settings_payload)
    payload: dict[str, Any] = {
        "status": "ACQUISITION_IN_PROGRESS",
        "timestamp_utc": utc_now(),
        "left_camera_serial": LEFT_CAMERA_SERIAL,
        "left_camera_role": "eye_to_hand_overhead",
        "horizontal_camera_serial": HORIZONTAL_CAMERA_SERIAL,
        "second_camera_role": "eye_in_hand_wrist",
        "maximum_observed_sync_error_ms": None,
        "maximum_allowed_sync_error_ms": MAXIMUM_ALLOWED_SYNC_ERROR_MS,
        "read_only": True,
        "servo_bus_accessed": False,
        "requested_duration_s": args.duration,
        "video_fps": args.video_fps,
        "video_frames": "native_resolution_unannotated",
        "camera_settings_file": str(args.camera_settings.resolve()),
        "camera_settings_sha256": sha256_file(args.camera_settings),
    }
    atomic_write_json(audit_path, payload)

    timestamps: FrameTimestampLog | None = None
    started_recorders: list[object] = []
    failure: BaseException | None = None
    capture_window_start_ns: int | None = None
    capture_window_end_ns: int | None = None
    failures = BackgroundFailure()
    directshow = None
    daheng = None
    try:
        timestamps = FrameTimestampLog(paths["frame_timestamps"])
        directshow = DirectShowRecorder(
            paths["horizontal_video"], settings, timestamps, failures, args.video_fps,
        )
        daheng = DahengRecorder(
            paths["left_video"], settings, timestamps, failures, args.video_fps,
            args.sdk_root,
        )
        recorders = (daheng, directshow)
        for recorder in recorders:
            recorder.start()
            started_recorders.append(recorder)
        ready_deadline = time.monotonic() + args.camera_ready_timeout_s
        for recorder in recorders:
            remaining = max(0.0, ready_deadline - time.monotonic())
            if not recorder.wait_ready(remaining):
                raise RuntimeError("camera did not produce its first auditable frame in time")
            failures.raise_if_set()

        capture_window_start_ns = time.monotonic_ns()
        capture_deadline = time.monotonic() + args.duration
        while True:
            failures.raise_if_set()
            remaining = capture_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(remaining, 0.02))
        capture_window_end_ns = time.monotonic_ns()
    except BaseException as error:
        failure = error
    finally:
        for recorder in started_recorders:
            recorder.stop()
        if timestamps is not None:
            timestamps.close()
    if failure is None:
        try:
            failures.raise_if_set()
        except BaseException as error:
            failure = error

    if failure is None:
        assert daheng is not None and directshow is not None
        assert capture_window_start_ns is not None and capture_window_end_ns is not None
        try:
            midpoints = load_capture_midpoints(
                paths["frame_timestamps"], start_ns=capture_window_start_ns,
                end_ns=capture_window_end_ns,
            )
            metrics = calculate_sync_metrics(
                midpoints[LEFT_TIMESTAMP_CAMERA],
                midpoints[HORIZONTAL_TIMESTAMP_CAMERA],
            )
            validations: dict[str, dict[str, float | int]] = {}
            for recorder in (daheng, directshow):
                cadence = validate_recorder_cadence(
                    frame_count=recorder.frame_count,
                    first_capture_mid_ns=recorder.first_capture_mid_ns,
                    last_capture_mid_ns=recorder.last_capture_mid_ns,
                    nominal_fps=args.video_fps,
                )
                video = validate_video_file(
                    recorder.output, expected_frames=recorder.frame_count,
                    nominal_fps=args.video_fps,
                )
                validations[recorder.output.name] = {**cadence, **video}
            maximum_error_ms = float(metrics["maximum_observed_sync_error_ms"])
            payload.update({
                "status": (
                    "PASS" if maximum_error_ms <= MAXIMUM_ALLOWED_SYNC_ERROR_MS
                    else "FAIL"
                ),
                "timestamp_utc": utc_now(),
                "capture_window_duration_s": (
                    capture_window_end_ns - capture_window_start_ns
                ) / 1_000_000_000.0,
                "left_frame_count": daheng.frame_count,
                "horizontal_frame_count": directshow.frame_count,
                "sample_count": metrics["sample_count"],
                "sample_counts": {
                    "left_frames_in_analysis_window": len(
                        midpoints[LEFT_TIMESTAMP_CAMERA]
                    ),
                    "horizontal_frames_in_analysis_window": len(
                        midpoints[HORIZONTAL_TIMESTAMP_CAMERA]
                    ),
                    "nearest_midpoint_pairs": metrics["sample_count"],
                },
                "maximum_observed_sync_error_ms": maximum_error_ms,
                "p95_sync_error_ms": metrics["p95_sync_error_ms"],
                "pairing_method": (
                    "each left capture midpoint paired to the nearest horizontal "
                    "capture midpoint in host monotonic time"
                ),
                "camera_video_validation": validations,
                "artifacts": _existing_artifacts(paths),
            })
        except BaseException as error:
            failure = error

    if failure is not None:
        payload.update({
            "status": "FAIL",
            "timestamp_utc": utc_now(),
            "failure_type": type(failure).__name__,
            "failure_message": str(failure),
            "artifacts": _existing_artifacts(paths),
        })
    atomic_write_json(audit_path, payload)
    return audit_path, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help="audited capture-window duration in seconds (default: 5)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--audit", type=Path,
        help="JSON output path (default: OUTPUT_DIR/camera_sync_audit.json)",
    )
    parser.add_argument("--camera-settings", type=Path, default=DEFAULT_CAMERA_SETTINGS)
    parser.add_argument("--sdk-root", type=Path, default=Path(r"D:\GalaxySDK"))
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--camera-ready-timeout-s", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_path, payload = run_audit(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"camera synchronization audit failed: {error}", file=sys.stderr)
        return 1
    print(audit_path)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
