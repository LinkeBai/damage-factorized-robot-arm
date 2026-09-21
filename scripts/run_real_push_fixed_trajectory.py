"""Execute one audited Level-A fixed raw-tick Push trajectory.

The default mode is a hardware-free dry-run.  Real execution additionally
requires ``--execute`` and an exact operator acknowledgement.  This runner does
not create trajectories, infer missing waypoints, label task success, or append
the formal completed-trials sheet.
"""
from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from robotarm.deployment.monitored_work import run_monitored_work

from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    JOINT_NAMES,
    TICKS_PER_DEGREE,
    VALID_CONDITIONS,
    CommandEvent,
    SafetyEnvelope,
    build_alignment_events,
    interpolate_raw_waypoints,
    load_fixed_raw_trajectory,
    load_safety_envelope,
    locked_indices_for_condition,
    minimum_safe_command_interval_s,
    sha256_file,
    validate_camera_settings,
    validate_interpolated_events,
)
from robotarm.hardware.live_trial_monitor import LiveTrialMonitor  # noqa: E402
from robotarm.deployment.real_calibration import radians_to_ticks, ticks_to_radians  # noqa: E402
from robotarm.deployment.ipwm_receding_horizon import IPWMRecedingHorizonPlanner  # noqa: E402


ACKNOWLEDGEMENT = "I_HAVE_CLEARED_WORKSPACE_SUPPORTED_ARM_AND_TESTED_ESTOP"
DAHENG_SERIAL = "FDE23080341"
DIRECTSHOW_INDEX = 1
SERVO_IDS = (1, 2, 3, 4, 5)
ESTOP_SERVO_IDS = (1, 2, 3, 4, 5, 6)
ADDRESS_OPERATION_MODE = 33
ADDRESS_TORQUE_ENABLE = 40
ADDRESS_ACCELERATION = 41
ADDRESS_GOAL_POSITION = 42
ADDRESS_GOAL_SPEED = 46
ADDRESS_PRESENT_POSITION = 56
ADDRESS_PRESENT_VOLTAGE = 62
ADDRESS_PRESENT_TEMPERATURE = 63
ADDRESS_PRESENT_CURRENT = 69
TRIAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
CRITICAL_WRITE_ATTEMPTS = 3
TORQUE_OFF_ATTEMPTS = 4
BUS_RETRY_DELAY_S = 0.02
TRAJECTORY_GOAL_VERIFY_EVERY_DISPATCHES = 10
TRAJECTORY_GOAL_VERIFY_PERIOD_S = 0.5
TRAJECTORY_GOAL_READ_ATTEMPTS = 2
TRAJECTORY_GOAL_READ_RETRY_DELAY_S = 0.005
TRAJECTORY_GOAL_TIMEOUT_ACCOUNTING_MARGIN_S = 0.004
TRAJECTORY_GOAL_CORRECTION_ROUNDS = 3
TRAJECTORY_GOAL_CORRECTION_SETTLE_S = 0.05
STATIC_BATCH_SETTLE_S = 0.10
STATIC_BATCH_READ_ATTEMPTS = 2
STATIC_BATCH_READ_RETRY_DELAY_S = 0.01
STATIC_BATCH_CORRECTION_ROUNDS = 3
STARTUP_SUPPORT_STABILITY_WINDOW_S = 0.5
MAXIMUM_UNPOWERED_DRIFT_DEG = 1.0
TELEMETRY_READ_ATTEMPTS = 2
TELEMETRY_READ_RETRY_DELAY_S = 0.003


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def signed_u16(value: int) -> int:
    return value - 65536 if value >= 32768 else value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    # A fixed ``.tmp`` name can collide with a fast subsequent manifest write
    # or a transient Windows scanner handle.  Use a per-write name and retry
    # only the final atomic replace; the payload is never written in-place.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}."
        f"{time.monotonic_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class BackgroundFailure:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._error: BaseException | None = None

    def set(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def raise_if_set(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"camera recorder failed: {error}") from error


class FrameTimestampLog:
    FIELDS = (
        "camera",
        "frame_index",
        "grab_start_monotonic_ns",
        "grab_end_monotonic_ns",
        "capture_mid_monotonic_ns",
        "host_utc",
        "device_frame_id",
        "device_timestamp",
        "width",
        "height",
    )

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        self._lock = threading.Lock()

    def write(self, row: dict[str, object]) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def _optional_method_value(instance: object, name: str) -> object:
    try:
        method = getattr(instance, name)
        return method()
    except Exception:
        return ""


def _wait_for_capture_deadline(stop: threading.Event, deadline_ns: int) -> bool:
    """Wait interruptibly for a frame deadline; return false when stopping."""
    while not stop.is_set():
        remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000.0
        if remaining_s <= 0.0:
            return True
        stop.wait(min(remaining_s, 0.02))
    return False


def _next_capture_deadline_ns(grab_start_ns: int, video_fps: float) -> int:
    return grab_start_ns + round(1_000_000_000.0 / video_fps)


class DirectShowRecorder:
    def __init__(
        self,
        output: Path,
        settings: dict[str, float],
        timestamps: FrameTimestampLog,
        failures: BackgroundFailure,
        video_fps: float,
        frame_callback: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.output = output
        self.settings = settings
        self.timestamps = timestamps
        self.failures = failures
        self.video_fps = video_fps
        self.frame_callback = frame_callback
        self.frame_count = 0
        self.first_capture_mid_ns: int | None = None
        self.last_capture_mid_ns: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="directshow-recorder", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.failures.set(RuntimeError("DirectShow recorder did not stop"))

    def _run(self) -> None:
        capture = None
        writer = None
        frame_size: tuple[int, int] | None = None
        try:
            import cv2

            capture = cv2.VideoCapture(DIRECTSHOW_INDEX, cv2.CAP_DSHOW)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open DirectShow camera index {DIRECTSHOW_INDEX}")
            capture.set(cv2.CAP_PROP_EXPOSURE, self.settings["second_exposure"])
            capture.set(cv2.CAP_PROP_BRIGHTNESS, self.settings["second_brightness"])
            next_capture_ns = time.monotonic_ns()
            while not self._stop.is_set():
                if not _wait_for_capture_deadline(self._stop, next_capture_ns):
                    break
                grab_start = time.monotonic_ns()
                ok, frame = capture.read()
                grab_end = time.monotonic_ns()
                if not ok or frame is None:
                    if self._stop.is_set():
                        break
                    raise RuntimeError("DirectShow frame loss")
                height, width = frame.shape[:2]
                if writer is None:
                    frame_size = (width, height)
                    writer = cv2.VideoWriter(
                        str(self.output),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        self.video_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer {self.output}")
                elif (width, height) != frame_size:
                    raise RuntimeError(
                        f"DirectShow frame size changed from {frame_size} to {(width, height)}"
                    )
                writer.write(frame)
                if self.frame_callback is not None:
                    self.frame_callback("wrist", frame)
                capture_mid = (grab_start + grab_end) // 2
                if self.first_capture_mid_ns is None:
                    self.first_capture_mid_ns = capture_mid
                self.last_capture_mid_ns = capture_mid
                self.timestamps.write({
                    "camera": "directshow_index_1",
                    "frame_index": self.frame_count,
                    "grab_start_monotonic_ns": grab_start,
                    "grab_end_monotonic_ns": grab_end,
                    "capture_mid_monotonic_ns": capture_mid,
                    "host_utc": utc_now(),
                    "device_frame_id": "",
                    "device_timestamp": "",
                    "width": width,
                    "height": height,
                })
                self.frame_count += 1
                self._ready.set()
                # Anchor the next deadline to the actual grab start.  This
                # cannot burst to "catch up" after a delayed frame and keeps
                # AVI wall duration consistent with its declared FPS.
                next_capture_ns = _next_capture_deadline_ns(grab_start, self.video_fps)
        except BaseException as error:
            self.failures.set(error)
            self._ready.set()
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()


class DahengRecorder:
    def __init__(
        self,
        output: Path,
        settings: dict[str, float],
        timestamps: FrameTimestampLog,
        failures: BackgroundFailure,
        video_fps: float,
        sdk_root: Path,
        frame_callback: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.output = output
        self.settings = settings
        self.timestamps = timestamps
        self.failures = failures
        self.video_fps = video_fps
        self.sdk_root = sdk_root
        self.frame_callback = frame_callback
        self.frame_count = 0
        self.first_capture_mid_ns: int | None = None
        self.last_capture_mid_ns: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="daheng-recorder", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.failures.set(RuntimeError("Daheng recorder did not stop"))

    def _run(self) -> None:
        camera = None
        writer = None
        streaming = False
        frame_size: tuple[int, int] | None = None
        try:
            import cv2
            from scripts.audit_daheng_camera import configure_sdk

            configure_sdk(self.sdk_root)
            import gxipy as gx  # type: ignore

            manager = gx.DeviceManager()
            _, devices = manager.update_device_list(1500)
            if not any(device.get("sn") == DAHENG_SERIAL for device in devices):
                raise RuntimeError(f"Daheng SN {DAHENG_SERIAL} was not enumerated")
            camera = manager.open_device_by_sn(DAHENG_SERIAL)
            features = camera.get_remote_device_feature_control()
            features.get_enum_feature("TriggerMode").set("Off")
            features.get_enum_feature("BalanceWhiteAuto").set("Continuous")
            features.get_float_feature("ExposureTime").set(
                self.settings["daheng_exposure"]
            )
            features.get_float_feature("Gain").set(self.settings["daheng_gain"])
            camera.stream_on()
            streaming = True
            next_capture_ns = time.monotonic_ns()
            while not self._stop.is_set():
                if not _wait_for_capture_deadline(self._stop, next_capture_ns):
                    break
                grab_start = time.monotonic_ns()
                raw = camera.data_stream[0].get_image(1000)
                grab_end = time.monotonic_ns()
                if raw is None:
                    if self._stop.is_set():
                        break
                    raise RuntimeError("Daheng frame loss")
                rgb = raw.convert("RGB").get_numpy_array()
                if rgb is None:
                    raise RuntimeError("Daheng RGB conversion returned no frame")
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                height, width = frame.shape[:2]
                if writer is None:
                    frame_size = (width, height)
                    writer = cv2.VideoWriter(
                        str(self.output),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        self.video_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer {self.output}")
                elif (width, height) != frame_size:
                    raise RuntimeError(
                        f"Daheng frame size changed from {frame_size} to {(width, height)}"
                    )
                writer.write(frame)
                if self.frame_callback is not None:
                    self.frame_callback("overhead", frame)
                capture_mid = (grab_start + grab_end) // 2
                if self.first_capture_mid_ns is None:
                    self.first_capture_mid_ns = capture_mid
                self.last_capture_mid_ns = capture_mid
                self.timestamps.write({
                    "camera": f"daheng_sn_{DAHENG_SERIAL}",
                    "frame_index": self.frame_count,
                    "grab_start_monotonic_ns": grab_start,
                    "grab_end_monotonic_ns": grab_end,
                    "capture_mid_monotonic_ns": capture_mid,
                    "host_utc": utc_now(),
                    "device_frame_id": _optional_method_value(raw, "get_frame_id"),
                    "device_timestamp": _optional_method_value(raw, "get_timestamp"),
                    "width": width,
                    "height": height,
                })
                self.frame_count += 1
                self._ready.set()
                next_capture_ns = _next_capture_deadline_ns(grab_start, self.video_fps)
        except BaseException as error:
            self.failures.set(error)
            self._ready.set()
        finally:
            if writer is not None:
                writer.release()
            if camera is not None:
                if streaming:
                    try:
                        camera.stream_off()
                    except Exception:
                        pass
                try:
                    camera.close_device()
                except Exception:
                    pass


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def write_u8_verified(
    bus: object,
    servo_id: int,
    address: int,
    value: int,
    *,
    attempts: int = CRITICAL_WRITE_ATTEMPTS,
    retry_delay_s: float = BUS_RETRY_DELAY_S,
) -> list[dict[str, object]]:
    """Write one byte and require matching register readback."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    history: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        event: dict[str, object] = {"attempt": attempt}
        try:
            bus.write_u8(servo_id, address, value)
            event["write"] = "sent"
        except BaseException as error:
            event["write_error"] = _error_text(error)
        try:
            observed = int(bus.read_u8(servo_id, address))
            event["readback"] = observed
            history.append(event)
            if observed == value:
                return history
        except BaseException as error:
            event["read_error"] = _error_text(error)
            history.append(event)
        if attempt < attempts and retry_delay_s > 0.0:
            time.sleep(retry_delay_s)
    raise RuntimeError(
        f"verified u8 write failed: servo={servo_id}, address={address}, "
        f"value={value}, history={history}"
    )


def write_u16_verified(
    bus: object,
    servo_id: int,
    address: int,
    value: int,
    *,
    attempts: int = CRITICAL_WRITE_ATTEMPTS,
    retry_delay_s: float = BUS_RETRY_DELAY_S,
) -> list[dict[str, object]]:
    """Write one word and require matching register readback."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    history: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        event: dict[str, object] = {"attempt": attempt}
        try:
            bus.write_u16(servo_id, address, value)
            event["write"] = "sent"
        except BaseException as error:
            event["write_error"] = _error_text(error)
        try:
            observed = int(bus.read_u16(servo_id, address))
            event["readback"] = observed
            history.append(event)
            if observed == value:
                return history
        except BaseException as error:
            event["read_error"] = _error_text(error)
            history.append(event)
        if attempt < attempts and retry_delay_s > 0.0:
            time.sleep(retry_delay_s)
    raise RuntimeError(
        f"verified u16 write failed: servo={servo_id}, address={address}, "
        f"value={value}, history={history}"
    )


@dataclass(frozen=True)
class StaticRegisterExpectation:
    label: str
    servo_id: int
    address: int
    width_bits: int
    expected: int

    def __post_init__(self) -> None:
        if self.servo_id not in SERVO_IDS:
            raise ValueError(f"unsupported static-config servo ID: {self.servo_id}")
        if self.width_bits not in (8, 16):
            raise ValueError("static register width must be 8 or 16 bits")


def _write_static_register(bus: object, item: StaticRegisterExpectation) -> None:
    if item.width_bits == 8:
        bus.write_u8(item.servo_id, item.address, item.expected)
    else:
        bus.write_u16(item.servo_id, item.address, item.expected)


def _read_static_register(bus: object, item: StaticRegisterExpectation) -> int:
    if item.width_bits == 8:
        return int(bus.read_u8(item.servo_id, item.address))
    return int(bus.read_u16(item.servo_id, item.address))


def batch_write_and_verify_static_registers(
    bus: object,
    expectations: Sequence[StaticRegisterExpectation],
    *,
    phase: str,
    settle_s: float = STATIC_BATCH_SETTLE_S,
    read_attempts: int = STATIC_BATCH_READ_ATTEMPTS,
    read_retry_delay_s: float = STATIC_BATCH_READ_RETRY_DELAY_S,
    correction_rounds: int = STATIC_BATCH_CORRECTION_ROUNDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Write a complete static phase before any read, then verify after settle.

    Only confirmed mismatches are rewritten.  A register that remains
    unreadable after the bounded read attempts fails the phase immediately;
    unknown state is never treated as permission to issue a correction write.
    """
    items = tuple(expectations)
    if not items:
        raise ValueError("static register batch must not be empty")
    if len({item.label for item in items}) != len(items):
        raise ValueError("static register labels must be unique")
    if read_attempts < 1 or correction_rounds < 0:
        raise ValueError("static batch attempt counts must be valid")
    if not math.isfinite(settle_s) or settle_s < 0.0:
        raise ValueError("static batch settle_s must be finite and nonnegative")
    if (not math.isfinite(read_retry_delay_s)
            or read_retry_delay_s < 0.0):
        raise ValueError(
            "static batch read_retry_delay_s must be finite and nonnegative"
        )

    def write_batch(selected: Sequence[StaticRegisterExpectation]) -> list[dict[str, object]]:
        writes: list[dict[str, object]] = []
        for item in selected:
            write_event: dict[str, object] = {
                "label": item.label,
                "servo_id": item.servo_id,
                "address": item.address,
                "width_bits": item.width_bits,
                "expected": item.expected,
                "write_monotonic_ns": time.monotonic_ns(),
            }
            try:
                _write_static_register(bus, item)
                write_event["write"] = "sent"
            except Exception as error:
                write_event["write_error"] = _error_text(error)
            writes.append(write_event)
        return writes

    def settle_record() -> dict[str, object]:
        record: dict[str, object] = {
            "duration_s": settle_s,
            "start_monotonic_ns": time.monotonic_ns(),
        }
        if settle_s > 0.0:
            sleep_fn(settle_s)
        record["end_monotonic_ns"] = time.monotonic_ns()
        return record

    def read_batch(
        selected: Sequence[StaticRegisterExpectation],
    ) -> tuple[list[dict[str, object]], list[str], str | None]:
        readbacks: list[dict[str, object]] = []
        mismatches: list[str] = []
        for item in selected:
            attempts_history: list[dict[str, object]] = []
            observed: int | None = None
            for attempt in range(1, read_attempts + 1):
                attempt_event: dict[str, object] = {"attempt": attempt}
                try:
                    observed = _read_static_register(bus, item)
                    attempt_event["value"] = observed
                    attempts_history.append(attempt_event)
                    break
                except Exception as error:
                    attempt_event["error"] = _error_text(error)
                    attempts_history.append(attempt_event)
                    if attempt < read_attempts and read_retry_delay_s > 0.0:
                        sleep_fn(read_retry_delay_s)
            read_event: dict[str, object] = {
                "label": item.label,
                "servo_id": item.servo_id,
                "address": item.address,
                "width_bits": item.width_bits,
                "expected": item.expected,
                "observed": observed,
                "attempts": attempts_history,
            }
            if observed is None:
                read_event["status"] = "PERSISTENT_READ_TIMEOUT"
                readbacks.append(read_event)
                return readbacks, mismatches, item.label
            if observed != item.expected:
                read_event["status"] = "MISMATCH"
                mismatches.append(item.label)
            else:
                read_event["status"] = "MATCH"
            readbacks.append(read_event)
        return readbacks, mismatches, None

    report: dict[str, object] = {
        "phase": phase,
        "timestamp_utc": utc_now(),
        "settle_s": settle_s,
        "read_attempt_limit": read_attempts,
        "read_retry_delay_s": read_retry_delay_s,
        "correction_round_limit": correction_rounds,
        "expectations": [
            {
                "label": item.label,
                "servo_id": item.servo_id,
                "address": item.address,
                "width_bits": item.width_bits,
                "expected": item.expected,
            }
            for item in items
        ],
        "initial_writes": write_batch(items),
        "initial_settle": None,
        "initial_readbacks": [],
        "correction_history": [],
    }
    report["initial_settle"] = settle_record()
    initial_readbacks, pending, timeout_label = read_batch(items)
    report["initial_readbacks"] = initial_readbacks
    report["original_mismatch_labels"] = list(pending)
    if timeout_label is not None:
        report.update({
            "status": "FAIL",
            "failure_code": "static_register_read_timeout",
            "failure_label": timeout_label,
        })
        return report
    if not pending:
        report["status"] = "PASS"
        return report

    by_label = {item.label: item for item in items}
    correction_history: list[dict[str, object]] = report["correction_history"]  # type: ignore[assignment]
    for round_index in range(1, correction_rounds + 1):
        targeted = [by_label[label] for label in pending]
        correction: dict[str, object] = {
            "round": round_index,
            "targeted_labels": list(pending),
            "writes": write_batch(targeted),
            "settle": None,
            "readbacks": [],
        }
        correction["settle"] = settle_record()
        readbacks, pending, timeout_label = read_batch(targeted)
        correction["readbacks"] = readbacks
        correction["unresolved_labels"] = [
            *pending, *([] if timeout_label is None else [timeout_label])
        ]
        correction_history.append(correction)
        if timeout_label is not None:
            report.update({
                "status": "FAIL",
                "failure_code": "static_register_read_timeout",
                "failure_label": timeout_label,
            })
            return report
        if not pending:
            report.update({
                "status": "CORRECTED_AFTER_RETRY",
                "correction_rounds_used": round_index,
            })
            return report

    report.update({
        "status": "FAIL",
        "failure_code": "static_register_mismatch",
        "failure_label": pending[0],
        "failure_labels": list(pending),
    })
    return report


def torque_off_all(
    bus: object,
    *,
    attempts: int = TORQUE_OFF_ATTEMPTS,
    retry_delay_s: float = BUS_RETRY_DELAY_S,
    settle_s: float = 0.02,
) -> dict[str, object]:
    """Immediately sweep IDs 1-6, then retry and read back every actuator.

    The first pass contains only writes so a slow or failing read from one servo
    cannot delay the initial torque-off request to later IDs.  Subsequent rounds
    reissue the request for every actuator not yet observed at zero.
    """
    if not math.isfinite(settle_s) or not 0 <= settle_s <= 0.02:
        raise ValueError("torque-off readback settling must be in [0, 0.02] seconds")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    records: dict[int, dict[str, object]] = {
        servo_id: {"write_attempts": [], "read_attempts": []}
        for servo_id in ESTOP_SERVO_IDS
    }
    pending = set(ESTOP_SERVO_IDS)
    for round_index in range(1, attempts + 1):
        # An all-ID write sweep is always performed before any readback.
        for servo_id in tuple(sorted(pending)):
            write_event: dict[str, object] = {"round": round_index}
            try:
                bus.write_u8(servo_id, ADDRESS_TORQUE_ENABLE, 0)
                write_event["write"] = "sent"
            except BaseException as error:
                write_event["error"] = _error_text(error)
            records[servo_id]["write_attempts"].append(write_event)  # type: ignore[union-attr]
        # All stop writes have already been sent. A measured bus turnaround
        # interval prevents the first read being lost during the write burst;
        # it never delays issuing the initial stop to any actuator.
        if settle_s:
            try:
                time.sleep(settle_s)
            except BaseException:
                pass
        verified_this_round: set[int] = set()
        for servo_id in tuple(sorted(pending)):
            read_event: dict[str, object] = {"round": round_index}
            try:
                observed = int(bus.read_u8(servo_id, ADDRESS_TORQUE_ENABLE))
                read_event["value"] = observed
                if observed == 0:
                    verified_this_round.add(servo_id)
            except BaseException as error:
                read_event["error"] = _error_text(error)
            records[servo_id]["read_attempts"].append(read_event)  # type: ignore[union-attr]
        pending -= verified_this_round
        if not pending:
            break
        if round_index < attempts and retry_delay_s > 0.0:
            try:
                time.sleep(retry_delay_s)
            except BaseException:
                # A repeated operator interrupt must not cancel the bounded
                # emergency shutdown sequence between its write sweeps.
                pass
    verified_ids = sorted(set(ESTOP_SERVO_IDS) - pending)
    enabled_ids: list[int] = []
    uncertain_ids: list[int] = []
    for servo_id in sorted(pending):
        reads = records[servo_id]["read_attempts"]
        observed = [event.get("value") for event in reads if "value" in event]  # type: ignore[union-attr]
        if observed and observed[-1] != 0:
            enabled_ids.append(servo_id)
        else:
            uncertain_ids.append(servo_id)
    status = (
        "VERIFIED_OFF"
        if not pending
        else "NOT_VERIFIED_OFF"
    )
    return {
        "status": status,
        "attempt_limit": attempts,
        "post_write_readback_settle_s": settle_s,
        "verified_ids": verified_ids,
        "enabled_ids": enabled_ids,
        "uncertain_ids": uncertain_ids,
        "servos": {str(key): value for key, value in records.items()},
    }


def record_torque_shutdown_report(
    manifest: dict[str, Any],
    reports: list[dict[str, object]],
    *,
    phase: str,
    report: dict[str, object],
) -> dict[str, object]:
    tagged = {"phase": phase, "timestamp_utc": utc_now(), **report}
    reports.append(tagged)
    latest_verified = tagged["status"] == "VERIFIED_OFF"
    earlier_verified = any(
        item.get("status") == "VERIFIED_OFF" for item in reports[:-1]
    )
    if latest_verified:
        overall = "VERIFIED_OFF"
    elif earlier_verified:
        overall = "LATEST_NOT_VERIFIED_AFTER_EARLIER_VERIFIED_OFF"
    else:
        overall = "NOT_VERIFIED_OFF"
    manifest.update({
        "torque_shutdown_status": overall,
        "torque_shutdown_latest_status": tagged["status"],
        "torque_shutdown_not_verified": not latest_verified,
        "torque_shutdown_readback_uncertain": bool(tagged.get("uncertain_ids")),
        "torque_shutdown_attempts": reports,
    })
    return tagged


@dataclass
class GoalValidationSchedule:
    every_dispatches: int
    period_s: float
    last_validation_s: float
    dispatches_since_validation: int = 0

    def __post_init__(self) -> None:
        if self.every_dispatches < 1:
            raise ValueError("every_dispatches must be positive")
        if not math.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError("period_s must be finite and positive")

    def note_dispatch(self) -> None:
        self.dispatches_since_validation += 1

    def due_reason(self, now_s: float) -> str | None:
        if self.dispatches_since_validation >= self.every_dispatches:
            return "dispatch_count"
        if now_s - self.last_validation_s >= self.period_s:
            return "time_period"
        return None

    def mark_validated(self, now_s: float) -> None:
        self.last_validation_s = now_s
        self.dispatches_since_validation = 0

    @staticmethod
    def final_reason() -> str:
        # Final validation is unconditional, even when a periodic validation
        # happened on the immediately preceding microstep.
        return "final_event"


def validate_trajectory_goal_readback(
    bus: object,
    expected_targets: Sequence[int],
    *,
    reason: str,
    dispatch_count: int,
    attempts: int = TRAJECTORY_GOAL_READ_ATTEMPTS,
    retry_delay_s: float = TRAJECTORY_GOAL_READ_RETRY_DELAY_S,
    correction_rounds: int = TRAJECTORY_GOAL_CORRECTION_ROUNDS,
    correction_settle_s: float = TRAJECTORY_GOAL_CORRECTION_SETTLE_S,
) -> dict[str, object]:
    """Batch-read goals and repair only axes with a confirmed mismatch.

    A persistent register read timeout is a feedback-loss event, not a repair
    opportunity.  It returns immediately without reading later axes or issuing
    any correction writes so the caller can enter the emergency shutdown path
    within the configured communication timeout.
    """
    if len(expected_targets) != len(SERVO_IDS):
        raise ValueError("expected_targets must contain five raw ticks")
    if attempts < 1 or correction_rounds < 0:
        raise ValueError("attempt counts must be valid")
    if not math.isfinite(retry_delay_s) or retry_delay_s < 0.0:
        raise ValueError("retry_delay_s must be finite and nonnegative")
    if not math.isfinite(correction_settle_s) or correction_settle_s < 0.0:
        raise ValueError("correction_settle_s must be finite and nonnegative")
    expected_values = tuple(int(value) for value in expected_targets)

    def read_all_axes() -> tuple[dict[str, object], list[str], str | None]:
        axes: dict[str, object] = {}
        mismatches: list[str] = []
        for servo_id, joint_name, expected in zip(
            SERVO_IDS, JOINT_NAMES, expected_values
        ):
            read_attempts: list[dict[str, object]] = []
            observed: int | None = None
            for attempt in range(1, attempts + 1):
                item: dict[str, object] = {"attempt": attempt}
                try:
                    observed = int(bus.read_u16(servo_id, ADDRESS_GOAL_POSITION))
                    item["value"] = observed
                    read_attempts.append(item)
                    break
                except BaseException as error:
                    item["error"] = _error_text(error)
                    read_attempts.append(item)
                    if attempt < attempts and retry_delay_s > 0.0:
                        time.sleep(retry_delay_s)
            axis: dict[str, object] = {
                "servo_id": servo_id,
                "expected_raw": expected,
                "observed_raw": observed,
                "read_attempts": read_attempts,
            }
            if observed is None:
                axis["status"] = "PERSISTENT_READ_TIMEOUT"
                axes[joint_name] = axis
                return axes, mismatches, joint_name
            elif observed != expected:
                axis["status"] = "MISMATCH"
                mismatches.append(joint_name)
            else:
                axis["status"] = "MATCH"
            axes[joint_name] = axis
        return axes, mismatches, None

    def fail_event(
        event: dict[str, object],
        *,
        failure_code: str,
        failure_joints: Sequence[str],
        timeout_axis: str | None = None,
        mismatch_axes: Sequence[str] = (),
    ) -> dict[str, object]:
        failure_list = list(failure_joints)
        event.update({
            "status": "FAIL",
            "failure_code": failure_code,
            "failure_joint": failure_list[0],
            "failure_joints": failure_list,
            "timeout_axes": [] if timeout_axis is None else [timeout_axis],
            "mismatch_axes": list(mismatch_axes),
            "read_end_monotonic_ns": time.monotonic_ns(),
        })
        return event

    event: dict[str, object] = {
        "reason": reason,
        "dispatch_count": dispatch_count,
        "timestamp_utc": utc_now(),
        "read_start_monotonic_ns": time.monotonic_ns(),
        "expected_targets_raw": list(expected_values),
        "correction_round_limit": correction_rounds,
        "correction_settle_s": correction_settle_s,
        "correction_history": [],
    }
    initial_axes, pending, timeout_axis = read_all_axes()
    event["axes"] = initial_axes
    event["original_problem_axes"] = [
        *pending, *([] if timeout_axis is None else [timeout_axis])
    ]
    if timeout_axis is not None:
        return fail_event(
            event,
            failure_code="goal_readback_timeout",
            failure_joints=[timeout_axis],
            timeout_axis=timeout_axis,
            mismatch_axes=pending,
        )
    if not pending:
        event["status"] = "PASS"
        event["read_end_monotonic_ns"] = time.monotonic_ns()
        return event

    correction_history: list[dict[str, object]] = event["correction_history"]  # type: ignore[assignment]
    name_to_index = {name: index for index, name in enumerate(JOINT_NAMES)}
    for round_index in range(1, correction_rounds + 1):
        targeted = list(pending)
        correction: dict[str, object] = {
            "round": round_index,
            "targeted_axes": targeted,
            "writes": [],
            "settle_s": correction_settle_s,
        }
        writes: list[dict[str, object]] = correction["writes"]  # type: ignore[assignment]
        for joint_name in targeted:
            index = name_to_index[joint_name]
            write_item: dict[str, object] = {
                "joint": joint_name,
                "servo_id": SERVO_IDS[index],
                "expected_raw": expected_values[index],
            }
            try:
                bus.write_u16(
                    SERVO_IDS[index], ADDRESS_GOAL_POSITION, expected_values[index]
                )
                write_item["write"] = "sent"
            except BaseException as error:
                write_item["error"] = _error_text(error)
            writes.append(write_item)
        correction_dispatch_ns = time.monotonic_ns()
        correction["dispatch_monotonic_ns"] = correction_dispatch_ns
        event["last_correction_dispatch_monotonic_ns"] = correction_dispatch_ns
        settle_started_ns = time.monotonic_ns()
        if correction_settle_s > 0.0:
            time.sleep(correction_settle_s)
        correction["settle_start_monotonic_ns"] = settle_started_ns
        correction["settle_end_monotonic_ns"] = time.monotonic_ns()
        readback_axes, pending, timeout_axis = read_all_axes()
        correction["readback_axes"] = readback_axes
        correction["unresolved_axes"] = [
            *pending, *([] if timeout_axis is None else [timeout_axis])
        ]
        correction_history.append(correction)
        if timeout_axis is not None:
            return fail_event(
                event,
                failure_code="goal_readback_timeout",
                failure_joints=[timeout_axis],
                timeout_axis=timeout_axis,
                mismatch_axes=pending,
            )
        if not pending:
            event.update({
                "status": "CORRECTED_AFTER_RETRY",
                "correction_rounds_used": round_index,
                "corrected_axes": list(event["original_problem_axes"]),
                "read_end_monotonic_ns": time.monotonic_ns(),
            })
            return event

    return fail_event(
        event,
        failure_code="goal_readback_mismatch",
        failure_joints=pending,
        mismatch_axes=pending,
    )


def read_u16_bounded(
    bus: object,
    servo_id: int,
    address: int,
    *,
    attempts: int = TRAJECTORY_GOAL_READ_ATTEMPTS,
    retry_delay_s: float = BUS_RETRY_DELAY_S,
) -> tuple[int | None, list[dict[str, object]]]:
    history: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        item: dict[str, object] = {"attempt": attempt}
        try:
            value = int(bus.read_u16(servo_id, address))
            item["value"] = value
            history.append(item)
            return value, history
        except BaseException as error:
            item["error"] = _error_text(error)
            history.append(item)
            if attempt < attempts and retry_delay_s > 0.0:
                time.sleep(retry_delay_s)
    return None, history


def build_damage_activation_record(
    *,
    condition: str,
    locked_index: int | None,
    target_raw: int | None,
    feedback_raw: int | None,
    feedback_read_attempts: list[dict[str, object]],
    maximum_drift_deg: float,
) -> dict[str, object]:
    if locked_index is None:
        return {
            "status": "NOT_APPLICABLE_INTACT",
            "condition": condition,
            "activation_phase": "not_applicable_intact",
        }
    record: dict[str, object] = {
        "condition": condition,
        "activation_phase": "after_start_alignment_before_fixed_trajectory",
        "locked_joint": JOINT_NAMES[locked_index],
        "locked_servo_id": SERVO_IDS[locked_index],
        "target_raw": target_raw,
        "feedback_raw": feedback_raw,
        "feedback_read_attempts": feedback_read_attempts,
        "maximum_allowed_feedback_error_deg": maximum_drift_deg,
        "timestamp_utc": utc_now(),
    }
    if feedback_raw is None or target_raw is None:
        record.update({
            "status": "FAIL",
            "failure_code": "alignment_lock_feedback_timeout",
        })
        return record
    error_ticks = abs(int(feedback_raw) - int(target_raw))
    error_deg = error_ticks / TICKS_PER_DEGREE
    record.update({
        "feedback_error_ticks": error_ticks,
        "feedback_error_deg": error_deg,
    })
    if error_deg > maximum_drift_deg:
        record.update({
            "status": "FAIL",
            "failure_code": "alignment_lock_error_exceeded",
        })
    else:
        record.update({
            "status": "PASS_DAMAGE_ACTIVE",
            "activated": True,
        })
    return record


def validate_recorder_cadence(
    *, frame_count: int, first_capture_mid_ns: int | None,
    last_capture_mid_ns: int | None, nominal_fps: float,
    tolerance_fraction: float = 0.15,
    absolute_tolerance_fps: float = 0.01,
    enforce_gate: bool = True,
) -> dict[str, float | int | str | bool]:
    if frame_count < 2 or first_capture_mid_ns is None or last_capture_mid_ns is None:
        raise RuntimeError("camera recording requires at least two timestamped frames")
    span_s = (last_capture_mid_ns - first_capture_mid_ns) / 1_000_000_000.0
    if span_s <= 0.0:
        raise RuntimeError("camera frame timestamps are not strictly increasing")
    observed_fps = (frame_count - 1) / span_s
    lower = nominal_fps * (1.0 - tolerance_fraction)
    upper = nominal_fps * (1.0 + tolerance_fraction)
    within_gate = not (
        observed_fps < lower - absolute_tolerance_fps
        or observed_fps > upper + absolute_tolerance_fps
    )
    if not within_gate and enforce_gate:
        raise RuntimeError(
            f"camera cadence {observed_fps:.3f} fps is inconsistent with "
            f"declared {nominal_fps:g} fps"
        )
    return {
        "frame_count": frame_count,
        "timestamp_span_s": span_s,
        "observed_fps": observed_fps,
        "lower_gate_fps": lower,
        "upper_gate_fps": upper,
        "absolute_numerical_tolerance_fps": absolute_tolerance_fps,
        "encoded_center_span_s": (frame_count - 1) / nominal_fps,
        "cadence_within_nominal_gate": within_gate,
        "cadence_policy": "strict" if enforce_gate else "record_only",
        "cadence_status": "PASS" if within_gate else "WARNING_RECORDED",
    }


def validate_video_file(
    path: Path, *, expected_frames: int, nominal_fps: float,
) -> dict[str, float | int]:
    import cv2

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"camera artifact is empty: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"recorded video is not decodable: {path}")
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, first = capture.read()
        if not ok or first is None:
            raise RuntimeError(f"recorded video contains no decodable frame: {path}")
    finally:
        capture.release()
    if abs(reported_frames - expected_frames) > 1:
        raise RuntimeError(
            f"video frame count mismatch for {path.name}: writer={expected_frames}, "
            f"decoded={reported_frames}"
        )
    if not math.isfinite(reported_fps) or abs(reported_fps - nominal_fps) > 0.1:
        raise RuntimeError(
            f"video FPS mismatch for {path.name}: expected={nominal_fps:g}, "
            f"decoded={reported_fps:g}"
        )
    return {
        "decoded_frame_count": reported_frames,
        "declared_fps": reported_fps,
        "width": width,
        "height": height,
    }


def read_positions(bus: object) -> tuple[int, int, int, int, int]:
    return tuple(
        int(bus.read_u16(servo_id, ADDRESS_PRESENT_POSITION))
        for servo_id in SERVO_IDS
    )  # type: ignore[return-value]


def goal_seed_expectations(
    targets: Sequence[int],
) -> tuple[StaticRegisterExpectation, ...]:
    if len(targets) != len(SERVO_IDS):
        raise ValueError("goal seed must contain five raw ticks")
    return tuple(
        StaticRegisterExpectation(
            label=f"{joint}.goal_position",
            servo_id=servo_id,
            address=ADDRESS_GOAL_POSITION,
            width_bits=16,
            expected=int(target),
        )
        for joint, servo_id, target in zip(JOINT_NAMES, SERVO_IDS, targets)
    )


def torque_off_static_configuration_expectations(
    targets: Sequence[int],
) -> tuple[StaticRegisterExpectation, ...]:
    goals = goal_seed_expectations(targets)
    accelerations = tuple(
        StaticRegisterExpectation(
            label=f"{joint}.acceleration",
            servo_id=servo_id,
            address=ADDRESS_ACCELERATION,
            width_bits=8,
            expected=1,
        )
        for joint, servo_id in zip(JOINT_NAMES, SERVO_IDS)
    )
    speeds = tuple(
        StaticRegisterExpectation(
            label=f"{joint}.goal_speed",
            servo_id=servo_id,
            address=ADDRESS_GOAL_SPEED,
            width_bits=16,
            expected=0,
        )
        for joint, servo_id in zip(JOINT_NAMES, SERVO_IDS)
    )
    # Register phases, rather than servo phases: every requested value is sent
    # before the common settle and the first readback.
    return (*goals, *accelerations, *speeds)


def torque_enable_expectations() -> tuple[StaticRegisterExpectation, ...]:
    return tuple(
        StaticRegisterExpectation(
            label=f"{joint}.torque_enable",
            servo_id=servo_id,
            address=ADDRESS_TORQUE_ENABLE,
            width_bits=8,
            expected=1,
        )
        for joint, servo_id in zip(JOINT_NAMES, SERVO_IDS)
    )


def require_static_batch_success(report: dict[str, object]) -> None:
    if report.get("status") not in ("PASS", "CORRECTED_AFTER_RETRY"):
        raise RuntimeError(
            "static register batch failed: "
            f"phase={report.get('phase')}, code={report.get('failure_code')}, "
            f"label={report.get('failure_label')}"
        )


def seed_and_enable_servos(
    bus: object,
    initial_targets: Sequence[int],
    *,
    settle_s: float = STATIC_BATCH_SETTLE_S,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compatibility helper using separate all-axis seed and torque batches."""
    modes = [int(bus.read_u8(servo_id, ADDRESS_OPERATION_MODE)) for servo_id in SERVO_IDS]
    if modes != [0, 0, 0, 0, 0]:
        raise RuntimeError(f"all positioning servos must already be in mode 0; got {modes}")
    seed_report = batch_write_and_verify_static_registers(
        bus,
        goal_seed_expectations(initial_targets),
        phase="latest_present_goal_seed",
        settle_s=settle_s,
        sleep_fn=sleep_fn,
    )
    require_static_batch_success(seed_report)
    torque_report = batch_write_and_verify_static_registers(
        bus,
        torque_enable_expectations(),
        phase="torque_enable_last",
        settle_s=settle_s,
        sleep_fn=sleep_fn,
    )
    require_static_batch_success(torque_report)
    return seed_report, torque_report


def command_changed_trajectory_targets(
    bus: object, previous: Sequence[int], targets: Sequence[int],
) -> tuple[int, ...]:
    """Write changed microstep axes only; periodic batch readback is separate."""
    if len(previous) != len(SERVO_IDS) or len(targets) != len(SERVO_IDS):
        raise ValueError("trajectory targets must contain five raw ticks")
    changed: list[int] = []
    writes = []
    for servo_id, before, target in zip(SERVO_IDS, previous, targets):
        if int(target) == int(before):
            continue
        # STS writes have no ACK.  Reading after every one-tick microstep
        # overloads the bus, so a bounded periodic all-axis verifier checks the
        # latest aggregate goal no later than 10 dispatches or 0.5 seconds.
        writes.append((servo_id, ADDRESS_GOAL_POSITION, int(target)))
        changed.append(servo_id)
    batch_writer = getattr(bus, "write_u16_batch", None)
    if callable(batch_writer):
        batch_writer(writes)
    else:
        for servo_id, address, value in writes:
            bus.write_u16(servo_id, address, value)
    return tuple(changed)


def telemetry_fields() -> list[str]:
    fields = [
        "sample_index", "host_utc", "read_start_monotonic_ns",
        "read_end_monotonic_ns", "elapsed_s", "phase", "segment_index",
        "read_retry_count",
    ]
    for name in JOINT_NAMES:
        fields.extend((
            f"{name}_position_raw", f"{name}_target_raw",
            f"{name}_position_rad", f"{name}_target_rad",
            f"{name}_voltage_raw", f"{name}_voltage_v",
            f"{name}_temperature_c", f"{name}_current_raw",
        ))
    return fields


def command_fields() -> list[str]:
    return [
        "command_index", "host_utc", "dispatch_monotonic_ns", "elapsed_s",
        "phase", "planned_phase_time_s", "segment_index", "changed_servo_ids",
        *[f"{name}_target_raw" for name in JOINT_NAMES],
    ]


def parse_xy_pair(value: str) -> tuple[float, float]:
    try:
        fields = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected x,y numeric pair") from error
    if len(fields) != 2 or not all(math.isfinite(field) for field in fields):
        raise argparse.ArgumentTypeError("expected two finite values formatted as x,y")
    return fields


def read_and_validate_telemetry(
    bus: object,
    *,
    target_raw: Sequence[int],
    phase: str,
    segment_index: int,
    run_started: float,
    sample_index: int,
    condition: str,
    safety: SafetyEnvelope,
    minimum_voltage_v: float,
) -> dict[str, object]:
    read_start = time.monotonic_ns()
    read_retry_count = 0

    def read_with_bounded_retry(reader: Callable[[], int]) -> int:
        nonlocal read_retry_count
        last_error: TimeoutError | None = None
        for attempt in range(TELEMETRY_READ_ATTEMPTS):
            try:
                return int(reader())
            except TimeoutError as error:
                last_error = error
                if attempt + 1 < TELEMETRY_READ_ATTEMPTS:
                    read_retry_count += 1
                    time.sleep(TELEMETRY_READ_RETRY_DELAY_S)
        assert last_error is not None
        raise last_error

    positions: list[int] = []
    voltages: list[int] = []
    temperatures: list[int] = []
    currents: list[int] = []
    for servo_id in SERVO_IDS:
        positions.append(read_with_bounded_retry(
            lambda servo_id=servo_id: bus.read_u16(servo_id, ADDRESS_PRESENT_POSITION)))
        voltages.append(read_with_bounded_retry(
            lambda servo_id=servo_id: bus.read_u8(servo_id, ADDRESS_PRESENT_VOLTAGE)))
        temperatures.append(read_with_bounded_retry(
            lambda servo_id=servo_id: bus.read_u8(servo_id, ADDRESS_PRESENT_TEMPERATURE)))
        currents.append(signed_u16(read_with_bounded_retry(
            lambda servo_id=servo_id: bus.read_u16(servo_id, ADDRESS_PRESENT_CURRENT))))
    read_end = time.monotonic_ns()
    voltage_v = [value / 10.0 for value in voltages]
    if min(voltage_v) < minimum_voltage_v:
        raise RuntimeError(
            f"undervoltage: minimum {min(voltage_v):.1f} V < {minimum_voltage_v:.1f} V"
        )
    if max(temperatures) >= safety.abort_temp_c:
        raise RuntimeError(
            f"temperature safety threshold exceeded: {max(temperatures)} C"
        )
    if max(abs(value) for value in currents) > safety.abort_current_raw:
        raise RuntimeError(
            f"raw-current safety threshold exceeded: {max(abs(value) for value in currents)}"
        )
    for position, joint in zip(positions, safety.joints):
        if position < joint.min_raw or position > joint.max_raw:
            raise RuntimeError(
                f"{joint.name} feedback {position} is outside measured raw limits "
                f"[{joint.min_raw}, {joint.max_raw}]"
            )
    for locked_index in locked_indices_for_condition(condition):
        drift_deg = abs(positions[locked_index] - target_raw[locked_index]) / TICKS_PER_DEGREE
        if drift_deg > safety.max_lock_drift_deg:
            raise RuntimeError(
                f"locked {JOINT_NAMES[locked_index]} drift {drift_deg:.3f} deg exceeds "
                f"{safety.max_lock_drift_deg:g} deg"
            )
    row: dict[str, object] = {
        "sample_index": sample_index,
        "host_utc": utc_now(),
        "read_start_monotonic_ns": read_start,
        "read_end_monotonic_ns": read_end,
        "elapsed_s": f"{time.monotonic() - run_started:.9f}",
        "phase": phase,
        "segment_index": segment_index,
        "read_retry_count": read_retry_count,
    }
    for index, name in enumerate(JOINT_NAMES):
        row.update({
            f"{name}_position_raw": positions[index],
            f"{name}_target_raw": int(target_raw[index]),
            f"{name}_position_rad": f"{math.radians(safety.joints[index].direction * (positions[index] - safety.joints[index].zero_raw) / TICKS_PER_DEGREE):.12f}",
            f"{name}_target_rad": f"{math.radians(safety.joints[index].direction * (int(target_raw[index]) - safety.joints[index].zero_raw) / TICKS_PER_DEGREE):.12f}",
            f"{name}_voltage_raw": voltages[index],
            f"{name}_voltage_v": f"{voltage_v[index]:.1f}",
            f"{name}_temperature_c": temperatures[index],
            f"{name}_current_raw": currents[index],
        })
    return row


def dry_run_payload(args: argparse.Namespace) -> tuple[dict[str, Any], Any, SafetyEnvelope]:
    safety = load_safety_envelope(args.safety)
    trajectory = load_fixed_raw_trajectory(
        args.waypoints,
        trajectory_id=args.trajectory_id,
        condition=args.condition,
        safety=safety,
        maximum_speed_deg_s=args.maximum_speed_deg_s,
    )
    events = interpolate_raw_waypoints(trajectory)
    actual_maximum = validate_interpolated_events(
        events, safety, args.maximum_speed_deg_s
    )
    if (trajectory.locked_joint_indices
            and args.maximum_lock_hold_s is not None
            and trajectory.duration_s > args.maximum_lock_hold_s):
        raise ValueError(
            f"locked trajectory duration {trajectory.duration_s:.3f}s exceeds "
            f"declared maximum lock hold {args.maximum_lock_hold_s:.3f}s"
        )
    camera_payload = json.loads(args.camera_settings.read_text(encoding="utf-8-sig"))
    camera_settings = validate_camera_settings(camera_payload)
    payload = {
        "status": "DRY_RUN_VALIDATED_NO_HARDWARE_ACCESSED",
        "trajectory_id": trajectory.trajectory_id,
        "condition": trajectory.condition,
        "waypoint_file": str(trajectory.source),
        "waypoint_sha256": trajectory.source_sha256,
        "waypoint_count": len(trajectory.waypoints),
        "interpolated_command_count": len(events),
        "duration_s": trajectory.duration_s,
        "maximum_waypoint_speed_deg_s": trajectory.maximum_commanded_speed_deg_s,
        "maximum_interpolated_speed_deg_s": actual_maximum,
        "maximum_allowed_speed_deg_s": args.maximum_speed_deg_s,
        "maximum_lock_hold_s": args.maximum_lock_hold_s,
        "locked_joint": trajectory.locked_joint_name,
        "locked_target_raw": (
            None if trajectory.locked_joint_index is None
            else trajectory.waypoints[0].targets_raw[trajectory.locked_joint_index]
        ),
        "locked_joints": [JOINT_NAMES[index] for index in trajectory.locked_joint_indices],
        "locked_targets_raw": {
            JOINT_NAMES[index]: trajectory.waypoints[0].targets_raw[index]
            for index in trajectory.locked_joint_indices
        },
        "camera_settings_file": str(args.camera_settings.resolve()),
        "camera_settings_sha256": sha256_file(args.camera_settings),
        "camera_settings": camera_settings,
        "daheng_white_balance_auto": "Continuous",
        "camera_devices": {
            "daheng_serial": DAHENG_SERIAL,
            "daheng_role": "eye_to_hand_overhead",
            "directshow_index": DIRECTSHOW_INDEX,
            "directshow_device": "USB VID_32E6 PID_9211",
            "directshow_role": "eye_in_hand_wrist",
        },
        "image_task": {
            "start_px": list(args.task_start_px),
            "goal_px": list(args.task_goal_px),
            "goal_gate_mode": args.task_goal_gate_mode,
            "goal_tolerance_px": args.task_goal_tolerance_px,
            "live_required_consecutive_frames": args.live_gate_consecutive_frames,
        },
        "notice": "validation only; this is not a physical trial or experimental result",
    }
    return payload, trajectory, safety


def execute_hardware(
    args: argparse.Namespace,
    dry_payload: dict[str, Any],
    trajectory: Any,
    safety: SafetyEnvelope,
) -> Path:
    if args.acknowledge_risk != ACKNOWLEDGEMENT:
        raise SystemExit(
            "refusing real motion: pass --acknowledge-risk " + ACKNOWLEDGEMENT
        )
    if not TRIAL_ID_PATTERN.fullmatch(args.trial_id or ""):
        raise SystemExit("--trial-id must be 1-80 safe filename characters")
    if not math.isfinite(args.telemetry_hz) or args.telemetry_hz <= 0.0:
        raise SystemExit("--telemetry-hz must be finite and positive")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0.0:
        raise SystemExit("--video-fps must be finite and positive")
    if not math.isfinite(args.camera_ready_timeout_s) or args.camera_ready_timeout_s <= 0.0:
        raise SystemExit("--camera-ready-timeout-s must be finite and positive")
    if (not math.isfinite(args.pre_roll_s) or args.pre_roll_s < 0.0
            or not math.isfinite(args.post_roll_s) or args.post_roll_s < 0.0):
        raise SystemExit("--pre-roll-s and --post-roll-s must be finite and nonnegative")
    if not 0.0 <= args.maximum_start_error_deg <= 5.0:
        raise SystemExit("--maximum-start-error-deg must be in [0, 5]")
    if not math.isfinite(args.minimum_voltage_v) or args.minimum_voltage_v <= 0.0:
        raise SystemExit("--minimum-voltage-v must be finite and positive")

    try:
        safety_document = yaml.safe_load(args.safety.read_text(encoding="utf-8-sig"))
        command_timeout_ms = float(
            safety_document["communications"]["command_timeout_ms"]
        )
    except (OSError, TypeError, KeyError, ValueError) as error:
        raise SystemExit(
            "safety file must define numeric communications.command_timeout_ms"
        ) from error
    if not math.isfinite(command_timeout_ms) or command_timeout_ms <= 0.0:
        raise SystemExit("communications.command_timeout_ms must be finite and positive")

    from scripts.recover_j5 import ServoBus

    online_planner = None
    online_archive = None
    online_axis = None
    online_plane_affine = None
    ipwm_online_archive_arg = getattr(args, "ipwm_online_archive", None)
    if ipwm_online_archive_arg is not None:
        # Load the frozen model and candidate family before cameras, serial I/O,
        # or damage activation.  GPU startup therefore cannot consume the
        # bounded physical lock interval.
        from scripts.prepare_real_ipwm_trial import (
            load_model as load_ipwm_model,
            score_references_terminal_only,
        )
        import torch
        online_archive = np.load(ipwm_online_archive_arg, allow_pickle=False)
        original_refs = np.asarray(online_archive["q_reference_rad"], dtype=float)
        if original_refs.shape[0] < 10_000:
            raise SystemExit("true IPWM loop requires at least 10,000 archived candidates")
        # The first commanded chunk must make progress.  For a 50-point bank
        # and H=5 this chooses points 9,19,29,39,49, not point zero repeatedly.
        sample_indices = np.linspace(
            original_refs.shape[1] / args.ipwm_online_horizon - 1,
            original_refs.shape[1] - 1,
            args.ipwm_online_horizon,
        ).round().astype(int)
        source_initial = np.asarray(online_archive["initial_state"], dtype=float)[:5]
        reference_deltas = original_refs[:, sample_indices, :] - source_initial[None, None, :]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise SystemExit("real online IPWM execution requires CUDA")
        ipwm_model = load_ipwm_model(
            args.ipwm_checkpoint, args.ipwm_model_config, device,
            shared_robot=getattr(args, "ipwm_shared_robot_fast_path", False),
        )
        def online_score(initial, candidates, goal, mask, angle):
            return score_references_terminal_only(
                ipwm_model, initial, candidates, goal, mask, angle,
                args.ipwm_online_score_batch_size,
            )
        online_axis = json.loads(args.ipwm_axis_calibration.read_text(encoding="utf-8"))
        diagnostics = online_axis["diagnostics"]
        if args.ipwm_plane_affine is not None:
            online_plane_affine = json.loads(args.ipwm_plane_affine.read_text(encoding="utf-8"))
            if online_plane_affine.get("status") != "PASS" or not online_plane_affine.get("inverse"):
                raise SystemExit("online 2-D plane affine must have status PASS and an inverse")
            inverse = online_plane_affine["inverse"]
            linear = np.asarray(inverse["pixel_to_base_linear_m_per_px"], dtype=float)
            offset = np.asarray(inverse["pixel_to_base_offset_m"], dtype=float)
            pixel_to_base_affine = np.column_stack((linear, offset))
        else:
            pixel_to_base_affine = None
        ranges = np.deg2rad(np.asarray([[j.min_deg, j.max_deg] for j in safety.joints]))
        if args.ipwm_max_j5_excursion_deg is not None:
            # This planner is constructed before the hardware/start-alignment
            # block initializes its local conversion arrays.  Derive the same
            # values here so the optional bound remains a pre-motion check.
            first_target_for_range = trajectory.waypoints[0].targets_raw
            zero_for_range = np.asarray([j.zero_raw for j in safety.joints], dtype=float)
            direction_for_range = np.asarray([j.direction for j in safety.joints], dtype=float)
            j5_center = ticks_to_radians(
                np.asarray(first_target_for_range, dtype=int),
                zero_for_range,
                direction_for_range,
            )[4]
            excursion = math.radians(args.ipwm_max_j5_excursion_deg)
            ranges[4, 0] = max(ranges[4, 0], j5_center - excursion)
            ranges[4, 1] = min(ranges[4, 1], j5_center + excursion)
        base_eligible = np.asarray(online_archive["selection_eligible"], dtype=bool)
        online_planner = IPWMRecedingHorizonPlanner(
            reference_deltas,
            task_start_px=args.task_start_px,
            task_goal_px=args.task_goal_px,
            locked_indices=locked_indices_for_condition(trajectory.condition),
            score_function=online_score,
            metres_per_pixel=float(diagnostics["metres_per_pixel"]),
            base_xy_per_pixel=np.asarray(diagnostics["base_xy_per_pixel"], dtype=float),
            base_xy_intercept_m=np.asarray(diagnostics["base_xy_intercept_m"], dtype=float),
            joint_ranges_rad=ranges,
            base_eligible=base_eligible,
            execution_reference_index=args.ipwm_online_execution_reference_index,
            minimum_remaining_fraction=args.ipwm_minimum_remaining_fraction,
            pixel_to_base_affine=pixel_to_base_affine,
        )

    trial_dir = (args.output_root / args.trial_id).resolve()
    if trial_dir.exists():
        raise SystemExit(f"refusing to overwrite existing trial directory: {trial_dir}")
    trial_dir.mkdir(parents=True)
    manifest_path = trial_dir / "run_manifest.json"
    powered_handoff = bool(getattr(args, "startup_powered_handoff", False))
    keep_powered = bool(getattr(args, "keep_torque_enabled_after_success", False))
    locked_indices = locked_indices_for_condition(trajectory.condition)
    if keep_powered and not powered_handoff:
        raise SystemExit("--keep-torque-enabled-after-success requires --startup-powered-handoff")
    manifest: dict[str, Any] = {
        **dry_payload,
        "status": "ACQUISITION_IN_PROGRESS",
        "trial_id": args.trial_id,
        "started_utc": utc_now(),
        "serial_port": args.port,
        "safety_file": str(args.safety.resolve()),
        "ipwm_plane_affine": (
            None if args.ipwm_plane_affine is None else {
                "path": str(args.ipwm_plane_affine.resolve()),
                "sha256": sha256_file(args.ipwm_plane_affine),
                "status": online_plane_affine.get("status") if online_plane_affine else None,
            }
        ),
        "safety_sha256": sha256_file(args.safety),
        "minimum_voltage_v": args.minimum_voltage_v,
        "maximum_start_error_deg": args.maximum_start_error_deg,
        "telemetry_hz": args.telemetry_hz,
        "telemetry_read_policy": {
            "attempts_per_register": TELEMETRY_READ_ATTEMPTS,
            "retry_delay_s": TELEMETRY_READ_RETRY_DELAY_S,
            "persistent_timeout": "abort_trial",
            "retry_count_column": "read_retry_count",
        },
        "static_register_batch_policy": {
            "initial_write_scope": "all_phase_registers_before_any_readback",
            "settle_s": STATIC_BATCH_SETTLE_S,
            "read_attempts": STATIC_BATCH_READ_ATTEMPTS,
            "read_retry_delay_s": STATIC_BATCH_READ_RETRY_DELAY_S,
            "confirmed_mismatch_correction_rounds": STATIC_BATCH_CORRECTION_ROUNDS,
            "correction_scope": "confirmed_mismatches_only",
            "persistent_read_timeout": "fail_without_correction",
            "phase_order": [
                "startup_verified_torque_off",
                "torque_off_goal_accel_speed_before_cameras",
                "camera_ready_and_pre_roll",
                "latest_present_goal_seed",
                "torque_enable_last",
            ],
        },
        "static_register_batch_events": [],
        "runtime_goal_read_attempts": TRAJECTORY_GOAL_READ_ATTEMPTS,
        "runtime_goal_read_retry_delay_s": TRAJECTORY_GOAL_READ_RETRY_DELAY_S,
        "communications_command_timeout_ms": command_timeout_ms,
        "torque_off_readback_attempts_per_shutdown": TORQUE_OFF_ATTEMPTS,
        "shutdown_phases": ["normal_completion_or_exception_immediate", "finally_repeat"],
        "trajectory_goal_readback_policy": {
            "microstep_write": "changed_joints_only_without_per_dispatch_readback",
            "verify_every_changed_dispatches": TRAJECTORY_GOAL_VERIFY_EVERY_DISPATCHES,
            "verify_period_s": TRAJECTORY_GOAL_VERIFY_PERIOD_S,
            "read_attempts_per_axis": TRAJECTORY_GOAL_READ_ATTEMPTS,
            "read_retry_delay_s": TRAJECTORY_GOAL_READ_RETRY_DELAY_S,
            "persistent_read_timeout": "fail_fast_without_correction_or_later_axis_reads",
            "verify_all_five_axes": True,
            "final_event_validation_required": True,
            "correction_enabled_for_confirmed_mismatch": True,
            "correction_round_limit": TRAJECTORY_GOAL_CORRECTION_ROUNDS,
            "correction_settle_s": TRAJECTORY_GOAL_CORRECTION_SETTLE_S,
            "correction_write_scope": "mismatched_axes_only",
            "correction_readback_scope": "all_five_axes_fail_fast",
            "corrected_event_status": "CORRECTED_AFTER_RETRY",
            "abort_on_unresolved_mismatch_or_persistent_timeout": True,
        },
        "trajectory_goal_validation_events": [],
        "damage_activation_phase": (
            "after_start_alignment_before_fixed_trajectory"
            if locked_indices else "not_applicable_intact"
        ),
        "task_trial_motion_phase": "fixed_trajectory",
        "start_alignment_counted_as_task_trial": False,
        "video_container": "AVI",
        "video_codec": "MJPG",
        "video_frames": "native_resolution_unannotated",
        "evidence_scope": "raw acquisition only; task outcome is unassessed",
        "startup_powered_handoff": powered_handoff,
        "keep_torque_enabled_after_success": keep_powered,
        "powered_hold_active": False,
        "planner_mode": "receding_horizon" if online_planner is not None else "open_loop_sequence",
        "ipwm_online_replan_cycles": [],
    }
    if online_planner is not None:
        manifest["ipwm_online_provenance"] = {
            "checkpoint": {"path": str(args.ipwm_checkpoint.resolve()),
                           "sha256": sha256_file(args.ipwm_checkpoint)},
            "model_config": {"path": str(args.ipwm_model_config.resolve()),
                             "sha256": sha256_file(args.ipwm_model_config)},
            "candidate_archive": {"path": str(ipwm_online_archive_arg.resolve()),
                                  "sha256": sha256_file(ipwm_online_archive_arg)},
            "axis_calibration": {"path": str(args.ipwm_axis_calibration.resolve()),
                                 "sha256": sha256_file(args.ipwm_axis_calibration)},
            "planner_source": {
                "path": str((ROOT / "src/robotarm/deployment/ipwm_receding_horizon.py").resolve()),
                "sha256": sha256_file(ROOT / "src/robotarm/deployment/ipwm_receding_horizon.py"),
            },
            "runner_source": {"path": str(Path(__file__).resolve()),
                              "sha256": sha256_file(Path(__file__))},
            "candidate_budget": int(len(online_archive["q_reference_rad"])),
            "horizon": int(args.ipwm_online_horizon),
            "shared_robot_fast_path": bool(getattr(args, "ipwm_shared_robot_fast_path", False)),
        }
    if online_planner is not None:
        # Preserve exact source bytes before acquisition; a hash of a mutable
        # worktree path alone is insufficient to reproduce a later trial.
        snapshot_root = trial_dir / "source_snapshot"
        snapshot_files = sorted((ROOT / "src").rglob("*.py"))
        snapshot_files += sorted((ROOT / "scripts").glob("*.py"))
        snapshot_files += [args.safety.resolve(), args.camera_settings.resolve(),
                           args.ipwm_model_config.resolve(), args.ipwm_axis_calibration.resolve()]
        manifest["source_snapshot"] = []
        for source in dict.fromkeys(snapshot_files):
            relative = source.resolve().relative_to(ROOT)
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["source_snapshot"].append({
                "path": str(destination.relative_to(trial_dir)),
                "original_path": str(source.resolve()),
                "sha256": sha256_file(destination),
            })
    atomic_write_json(manifest_path, manifest)

    timestamps = FrameTimestampLog(trial_dir / "frame_timestamps.csv")
    failures = BackgroundFailure()
    settings = dry_payload["camera_settings"]
    task_start_px = getattr(args, "task_start_px", (1102.34, 565.0))
    task_goal_px = getattr(args, "task_goal_px", (1132.34, 565.0))
    task_goal_tolerance_px = getattr(args, "task_goal_tolerance_px", 5.0)
    live_gate_consecutive_frames = getattr(args, "live_gate_consecutive_frames", 3)
    stop_on_live_goal = bool(getattr(args, "stop_on_live_goal", False))
    live_monitor = LiveTrialMonitor(
        start_px=task_start_px,
        task_goal_px=task_goal_px,
        goal_tolerance_px=task_goal_tolerance_px,
        gate_mode=getattr(args, "task_goal_gate_mode", "axiswise"),
        stable_goal_frames=live_gate_consecutive_frames,
    )
    live_monitor.start()
    directshow = DirectShowRecorder(
        trial_dir / "directshow_index1_raw.avi", settings, timestamps, failures,
        args.video_fps, live_monitor.publish,
    )
    daheng = DahengRecorder(
        trial_dir / f"daheng_{DAHENG_SERIAL}_raw.avi", settings, timestamps,
        failures, args.video_fps, args.sdk_root, live_monitor.publish,
    )
    recorders = (daheng, directshow)
    telemetry_handle = (trial_dir / "servo_telemetry.csv").open(
        "w", newline="", encoding="utf-8"
    )
    telemetry_writer = csv.DictWriter(telemetry_handle, fieldnames=telemetry_fields())
    telemetry_writer.writeheader()
    telemetry_handle.flush()
    command_handle = (trial_dir / "commands.csv").open("w", newline="", encoding="utf-8")
    command_writer = csv.DictWriter(command_handle, fieldnames=command_fields())
    command_writer.writeheader()
    command_handle.flush()

    bus = None
    started_recorders: list[object] = []
    telemetry_count = 0
    command_count = 0
    microstep_dispatch_count_total = 0
    start_alignment_dispatch_count = 0
    fixed_trajectory_dispatch_count = 0
    static_register_batch_events: list[dict[str, object]] = []
    manifest["static_register_batch_events"] = static_register_batch_events
    goal_validation_events: list[dict[str, object]] = []
    manifest["trajectory_goal_validation_events"] = goal_validation_events
    shutdown_reports: list[dict[str, object]] = []
    completed_with_powered_hold = False

    def request_and_record_shutdown(phase: str) -> dict[str, object]:
        if bus is None:
            raise RuntimeError("servo bus is not open")
        report = record_torque_shutdown_report(
            manifest,
            shutdown_reports,
            phase=phase,
            report=torque_off_all(bus),
        )
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        return report

    def record_static_register_batch(report: dict[str, object]) -> None:
        static_register_batch_events.append(report)
        manifest.update({
            "static_register_batch_events": static_register_batch_events,
            "static_register_batch_count": len(static_register_batch_events),
            "latest_static_register_batch_status": report.get("status"),
            "latest_static_register_batch_phase": report.get("phase"),
        })
        atomic_write_json(manifest_path, manifest)
        require_static_batch_success(report)

    try:
        bus = ServoBus(args.port)
        serial_read_timeout_s = float(bus.serial.timeout)
        servo_single_read_deadline_s = float(bus.READ_DEADLINE_S)
        runtime_goal_timeout_bound_s = (
            TRAJECTORY_GOAL_READ_ATTEMPTS * servo_single_read_deadline_s
            + (TRAJECTORY_GOAL_READ_ATTEMPTS - 1)
            * TRAJECTORY_GOAL_READ_RETRY_DELAY_S
            + TRAJECTORY_GOAL_TIMEOUT_ACCOUNTING_MARGIN_S
        )
        manifest.update({
            "serial_read_timeout_s": serial_read_timeout_s,
            "servo_single_read_deadline_s": servo_single_read_deadline_s,
            "runtime_goal_timeout_accounting_margin_s": (
                TRAJECTORY_GOAL_TIMEOUT_ACCOUNTING_MARGIN_S
            ),
            "runtime_goal_single_axis_timeout_bound_s": runtime_goal_timeout_bound_s,
        })
        if runtime_goal_timeout_bound_s * 1000.0 >= command_timeout_ms:
            raise RuntimeError(
                "runtime goal readback timeout bound "
                f"{runtime_goal_timeout_bound_s * 1000.0:.1f} ms is not strictly below "
                f"communications.command_timeout_ms={command_timeout_ms:g}"
            )
        atomic_write_json(manifest_path, manifest)

        # Default cold start begins torque-off.  An explicit powered handoff is
        # read-only at startup so a gravity-loaded arm cannot collapse between
        # recovery and acquisition.
        # Keep this separate from the emergency/final shutdown history so an
        # earlier off observation cannot mask an uncertain final shutdown.
        if powered_handoff:
            startup_torque = tuple(
                int(bus.read_u8(servo_id, ADDRESS_TORQUE_ENABLE))
                for servo_id in SERVO_IDS
            )
            manifest["startup_torque_enable_readback_ids_1_5"] = list(startup_torque)
            atomic_write_json(manifest_path, manifest)
            if startup_torque != (1, 1, 1, 1, 1):
                raise RuntimeError(
                    "powered handoff requires all five positioning joints enabled; "
                    f"got {startup_torque}"
                )
        else:
            startup_torque_off = torque_off_all(bus)
            manifest["startup_torque_off"] = startup_torque_off
            atomic_write_json(manifest_path, manifest)
            if startup_torque_off["status"] != "VERIFIED_OFF":
                raise RuntimeError(
                    "startup torque-off was not verified for every actuator: "
                    f"enabled={startup_torque_off['enabled_ids']}, "
                    f"uncertain={startup_torque_off['uncertain_ids']}"
                )

        modes = [
            int(bus.read_u8(servo_id, ADDRESS_OPERATION_MODE))
            for servo_id in SERVO_IDS
        ]
        if modes != [0, 0, 0, 0, 0]:
            raise RuntimeError(
                f"all positioning servos must already be in mode 0; got {modes}"
            )
        if powered_handoff:
            present = read_positions(bus)
        else:
            unpowered_present_before = read_positions(bus)
            time.sleep(STARTUP_SUPPORT_STABILITY_WINDOW_S)
            present = read_positions(bus)
            unpowered_drift_deg = [
                abs(after - before) / TICKS_PER_DEGREE
                for before, after in zip(unpowered_present_before, present)
            ]
            manifest.update({
                "unpowered_support_stability_window_s": STARTUP_SUPPORT_STABILITY_WINDOW_S,
                "unpowered_position_before_raw": list(unpowered_present_before),
                "unpowered_position_after_raw": list(present),
                "unpowered_drift_deg": unpowered_drift_deg,
                "maximum_allowed_unpowered_drift_deg": MAXIMUM_UNPOWERED_DRIFT_DEG,
            })
            atomic_write_json(manifest_path, manifest)
            if max(unpowered_drift_deg) > MAXIMUM_UNPOWERED_DRIFT_DEG:
                raise RuntimeError(
                    "mechanical support is not stable while torque is off: max drift "
                    f"{max(unpowered_drift_deg):.3f} deg > "
                    f"{MAXIMUM_UNPOWERED_DRIFT_DEG:g} deg"
                )
        first_target = trajectory.waypoints[0].targets_raw
        for position, joint in zip(present, safety.joints):
            if position < joint.min_raw or position > joint.max_raw:
                raise RuntimeError(
                    f"initial {joint.name} feedback {position} is outside measured raw "
                    f"limits [{joint.min_raw}, {joint.max_raw}]"
                )
        start_errors = [
            abs(actual - target) / TICKS_PER_DEGREE
            for actual, target in zip(present, first_target)
        ]
        if max(start_errors) > args.maximum_start_error_deg:
            raise RuntimeError(
                f"robot is not at trajectory start: max error {max(start_errors):.3f} deg "
                f"> {args.maximum_start_error_deg:g} deg"
            )
        manifest.update({
            "initial_position_raw": list(present),
            "first_waypoint_target_raw": list(first_target),
            "initial_error_deg": start_errors,
            "startup_operation_modes": modes,
            "static_configuration_torque_state": (
                "VERIFIED_ON_POWERED_HANDOFF" if powered_handoff else "VERIFIED_OFF"
            ),
        })
        atomic_write_json(manifest_path, manifest)

        # Configure all torque-off static registers before cameras.  Every
        # write in this phase precedes the common settle and first readback.
        if not powered_handoff:
            record_static_register_batch(batch_write_and_verify_static_registers(
                bus,
                torque_off_static_configuration_expectations(present),
                phase="torque_off_goal_accel_speed_before_cameras",
            ))

        for recorder in recorders:
            recorder.start()
            started_recorders.append(recorder)
        deadline = time.monotonic() + args.camera_ready_timeout_s
        for recorder in recorders:
            remaining = max(0.0, deadline - time.monotonic())
            if not recorder.wait_ready(remaining):
                raise RuntimeError("camera did not produce its first auditable frame in time")
            failures.raise_if_set()

        if args.pre_roll_s > 0.0:
            pre_roll_deadline = time.monotonic() + args.pre_roll_s
            while time.monotonic() < pre_roll_deadline:
                failures.raise_if_set()
                time.sleep(max(0.0, min(0.02, pre_roll_deadline - time.monotonic())))

        # The pre-camera snapshot is deliberately not reused here.  Read and
        # validate the latest free-arm pose after pre-roll, then seed all five
        # goals as one batch before torque-enable is attempted.
        latest_present = read_positions(bus)
        for position, joint in zip(latest_present, safety.joints):
            if position < joint.min_raw or position > joint.max_raw:
                raise RuntimeError(
                    f"latest {joint.name} feedback {position} is outside measured raw "
                    f"limits [{joint.min_raw}, {joint.max_raw}]"
                )
        latest_start_errors = [
            abs(actual - target) / TICKS_PER_DEGREE
            for actual, target in zip(latest_present, first_target)
        ]
        if max(latest_start_errors) > args.maximum_start_error_deg:
            raise RuntimeError(
                "robot moved away from the trajectory start before enable: max error "
                f"{max(latest_start_errors):.3f} deg > "
                f"{args.maximum_start_error_deg:g} deg"
            )
        manifest.update({
            "latest_pre_enable_position_raw": list(latest_present),
            "latest_pre_enable_error_deg": latest_start_errors,
        })
        atomic_write_json(manifest_path, manifest)

        if not powered_handoff:
            record_static_register_batch(batch_write_and_verify_static_registers(
                bus,
                goal_seed_expectations(latest_present),
                phase="latest_present_goal_seed",
            ))
            failures.raise_if_set()
            record_static_register_batch(batch_write_and_verify_static_registers(
                bus,
                torque_enable_expectations(),
                phase="torque_enable_last",
            ))

        # In D2/D3 the locked coordinate reaches its frozen value only via the
        # bounded alignment phase; damage is not active during alignment.
        initial_target = list(latest_present)
        run_started = time.monotonic()
        last_dispatch = run_started
        current_target = tuple(initial_target)
        telemetry_period = 1.0 / args.telemetry_hz
        next_telemetry = run_started
        sample_index = 0
        goal_schedule = GoalValidationSchedule(
            every_dispatches=TRAJECTORY_GOAL_VERIFY_EVERY_DISPATCHES,
            period_s=TRAJECTORY_GOAL_VERIFY_PERIOD_S,
            last_validation_s=run_started,
        )

        def log_command(
            *, phase: str, planned_time_s: float, segment_index: int,
            target: Sequence[int], dispatch_ns: int,
            changed_servo_ids: Sequence[int],
        ) -> None:
            nonlocal command_count
            row: dict[str, object] = {
                "command_index": command_count,
                "host_utc": utc_now(),
                "dispatch_monotonic_ns": dispatch_ns,
                "elapsed_s": f"{time.monotonic() - run_started:.9f}",
                "phase": phase,
                "planned_phase_time_s": f"{planned_time_s:.9f}",
                "segment_index": segment_index,
                "changed_servo_ids": ";".join(str(value) for value in changed_servo_ids),
            }
            row.update({f"{name}_target_raw": int(value)
                        for name, value in zip(JOINT_NAMES, target)})
            command_writer.writerow(row)
            command_handle.flush()
            command_count += 1

        log_command(
            phase="seed_hold", planned_time_s=0.0, segment_index=-1,
            target=current_target, dispatch_ns=time.monotonic_ns(),
            changed_servo_ids=SERVO_IDS,
        )

        def perform_goal_validation(reason: str) -> dict[str, object]:
            nonlocal last_dispatch
            event = validate_trajectory_goal_readback(
                bus,
                current_target,
                reason=reason,
                dispatch_count=microstep_dispatch_count_total,
            )
            goal_validation_events.append(event)
            manifest.update({
                "trajectory_goal_validation_events": goal_validation_events,
                "trajectory_goal_validation_count": len(goal_validation_events),
                "microstep_dispatch_count_total": microstep_dispatch_count_total,
                "start_alignment_dispatch_count": start_alignment_dispatch_count,
                "fixed_trajectory_dispatch_count": fixed_trajectory_dispatch_count,
            })
            try:
                atomic_write_json(manifest_path, manifest)
            except Exception:
                pass
            if event["status"] not in ("PASS", "CORRECTED_AFTER_RETRY"):
                raise RuntimeError(
                    f"trajectory goal validation failed: code={event.get('failure_code')}, "
                    f"joint={event.get('failure_joint')}, reason={reason}"
                )
            correction_dispatch_ns = event.get(
                "last_correction_dispatch_monotonic_ns"
            )
            if correction_dispatch_ns is not None:
                # Treat the repair batch as the most recent command dispatch.
                # This keeps the next microstep behind the same <=5 deg/s
                # inter-dispatch gate instead of catching up immediately.
                last_dispatch = max(
                    last_dispatch, int(correction_dispatch_ns) / 1_000_000_000.0
                )
            goal_schedule.mark_validated(time.monotonic())
            return event

        def monitor_until(deadline_s: float, *, phase: str, segment_index: int) -> None:
            nonlocal next_telemetry, sample_index, telemetry_count
            while True:
                failures.raise_if_set()
                if (trial_dir / "STOP_REQUESTED").exists():
                    raise RuntimeError("operator stop requested via trial STOP_REQUESTED file")
                now = time.monotonic()
                if (online_planner is not None and locked_indices
                        and damage_active_started is not None
                        and args.maximum_lock_hold_s is not None
                        and now - damage_active_started >= args.maximum_lock_hold_s):
                    raise RuntimeError("online IPWM lock-hold deadline exceeded")
                validation_reason = goal_schedule.due_reason(now)
                if validation_reason is not None:
                    perform_goal_validation(validation_reason)
                    now = time.monotonic()
                if now >= next_telemetry:
                    row = read_and_validate_telemetry(
                        bus,
                        target_raw=current_target,
                        phase=phase,
                        segment_index=segment_index,
                        run_started=run_started,
                        sample_index=sample_index,
                        condition=trajectory.condition,
                        safety=safety,
                        minimum_voltage_v=args.minimum_voltage_v,
                    )
                    telemetry_writer.writerow(row)
                    telemetry_handle.flush()
                    sample_index += 1
                    telemetry_count += 1
                    now = time.monotonic()
                    next_telemetry = max(next_telemetry + telemetry_period,
                                         now + min(0.001, telemetry_period))
                if now >= deadline_s:
                    return
                next_goal_validation = (
                    goal_schedule.last_validation_s + goal_schedule.period_s
                )
                time.sleep(min(
                    0.01,
                    deadline_s - now,
                    max(0.001, next_telemetry - now),
                    max(0.001, next_goal_validation - now),
                ))

        live_goal_stop_triggered = False
        live_goal_stop_reason: str | None = None
        first_live_goal_stop_trace: dict[str, object] | None = None
        stagnation_observations: list[dict[str, object]] = []
        damage_active_started: float | None = None

        def run_events(events: Sequence[CommandEvent], phase: str) -> None:
            nonlocal last_dispatch, current_target, microstep_dispatch_count_total
            nonlocal start_alignment_dispatch_count, fixed_trajectory_dispatch_count
            nonlocal live_goal_stop_triggered, live_goal_stop_reason
            nonlocal first_live_goal_stop_trace
            phase_started = time.monotonic()
            for event in events:
                if phase == "fixed_trajectory":
                    changed_locks = [
                        JOINT_NAMES[index] for index in locked_indices
                        if event.targets_raw[index] != first_target[index]
                    ]
                    if changed_locks:
                        raise RuntimeError(
                            "fixed trajectory attempted to change locked targets: "
                            + ",".join(changed_locks)
                        )
                scheduled = phase_started + event.time_s
                safe_interval = minimum_safe_command_interval_s(
                    current_target, event.targets_raw, safety, args.maximum_speed_deg_s
                )
                # A goal-validation repair may happen while monitor_until is
                # waiting.  It advances last_dispatch, so recompute the gate
                # until the original schedule and the post-repair speed gate
                # are both satisfied.
                while True:
                    if (online_planner is not None and locked_indices
                            and damage_active_started is not None
                            and args.maximum_lock_hold_s is not None
                            and time.monotonic() - damage_active_started
                            >= args.maximum_lock_hold_s):
                        raise RuntimeError("online IPWM lock-hold deadline exceeded")
                    due = max(scheduled, last_dispatch + safe_interval)
                    monitor_until(
                        due, phase=phase, segment_index=event.segment_index
                    )
                    stable_goal = live_monitor.stable_goal_reached()
                    crossed_goal_plane = live_monitor.goal_plane_crossed()
                    if (phase == "fixed_trajectory" and stop_on_live_goal
                            and (stable_goal or crossed_goal_plane)):
                        live_goal_stop_triggered = True
                        live_goal_stop_reason = (
                            "strict_radial_gate" if stable_goal else "goal_plane_guard"
                        )
                        if first_live_goal_stop_trace is None:
                            first_live_goal_stop_trace = {
                                "decision_monotonic_ns": time.monotonic_ns(),
                                "location": "before_microstep_dispatch",
                                "segment_index": event.segment_index,
                                "stable_goal": bool(stable_goal),
                                "goal_plane_crossed": bool(crossed_goal_plane),
                                "held_target_raw": list(current_target),
                                "cancelled_next_target_raw": list(event.targets_raw),
                            }
                        return
                    if time.monotonic() >= max(
                        scheduled, last_dispatch + safe_interval
                    ):
                        break
                failures.raise_if_set()
                changed_servo_ids: tuple[int, ...] = ()
                if tuple(event.targets_raw) != tuple(current_target):
                    changed_servo_ids = command_changed_trajectory_targets(
                        bus, current_target, event.targets_raw
                    )
                    current_target = event.targets_raw
                    last_dispatch = time.monotonic()
                    microstep_dispatch_count_total += 1
                    if phase == "start_alignment":
                        start_alignment_dispatch_count += 1
                    elif phase == "fixed_trajectory":
                        fixed_trajectory_dispatch_count += 1
                    goal_schedule.note_dispatch()
                    validation_reason = goal_schedule.due_reason(time.monotonic())
                    if validation_reason is not None:
                        perform_goal_validation(validation_reason)
                dispatch_ns = time.monotonic_ns()
                log_command(
                    phase=phase,
                    planned_time_s=event.time_s,
                    segment_index=event.segment_index,
                    target=current_target,
                    dispatch_ns=dispatch_ns,
                    changed_servo_ids=changed_servo_ids,
                )

        alignment = build_alignment_events(
            current_target, first_target, safety, args.maximum_speed_deg_s
        )
        manifest.update({
            "alignment_command_count": len(alignment),
            "alignment_planned_duration_s": alignment[-1].time_s if alignment else 0.0,
        })
        atomic_write_json(manifest_path, manifest)
        run_events(alignment, "start_alignment")
        alignment_goal_validation = perform_goal_validation("alignment_complete")
        if not locked_indices:
            damage_activation = build_damage_activation_record(
                condition=trajectory.condition,
                locked_index=None,
                target_raw=None,
                feedback_raw=None,
                feedback_read_attempts=[],
                maximum_drift_deg=safety.max_lock_drift_deg,
            )
        else:
            axis_records = []
            for locked_index in locked_indices:
                locked_feedback, locked_feedback_attempts = read_u16_bounded(
                    bus, SERVO_IDS[locked_index], ADDRESS_PRESENT_POSITION,
                )
                axis_record = build_damage_activation_record(
                    condition=trajectory.condition,
                    locked_index=locked_index,
                    target_raw=first_target[locked_index],
                    feedback_raw=locked_feedback,
                    feedback_read_attempts=locked_feedback_attempts,
                    maximum_drift_deg=safety.max_lock_drift_deg,
                )
                locked_goal_axis = alignment_goal_validation["axes"][JOINT_NAMES[locked_index]]
                axis_record.update({
                    "goal_readback_raw": locked_goal_axis["observed_raw"],
                    "goal_readback_status": locked_goal_axis["status"],
                    "goal_validation_event_index": len(goal_validation_events) - 1,
                })
                axis_records.append(axis_record)
            failures_by_axis = [row for row in axis_records if row["status"] == "FAIL"]
            damage_activation = {
                "status": "FAIL" if failures_by_axis else "PASS_DAMAGE_ACTIVE",
                "condition": trajectory.condition,
                "activation_phase": "after_start_alignment_before_fixed_trajectory",
                "atomic_batch_semantics": "all goals validated before task motion",
                "locked_axes": axis_records,
                "failure_code": (
                    "one_or_more_lock_axes_failed_activation" if failures_by_axis else None
                ),
            }
        manifest["damage_activation"] = damage_activation
        atomic_write_json(manifest_path, manifest)
        if damage_activation["status"] == "FAIL":
            raise RuntimeError(
                "damage activation failed after alignment: "
                f"{damage_activation.get('failure_code')}"
            )
        damage_active_started = time.monotonic()
        if online_planner is None:
            planned_events = interpolate_raw_waypoints(trajectory)
            run_events(planned_events, "fixed_trajectory")
        else:
            zero = np.asarray([j.zero_raw for j in safety.joints], dtype=float)
            direction = np.asarray([j.direction for j in safety.joints], dtype=float)
            previous_q = None
            previous_q_time = None
            last_vision_ns = -1
            fixed_lock_angles = ticks_to_radians(
                np.asarray(first_target, dtype=int), zero, direction
            )
            cycle_dir = trial_dir / "ipwm_replan_cycles"
            cycle_dir.mkdir()
            cycle_indices = (itertools.count() if args.ipwm_online_replans is None
                             else range(args.ipwm_online_replans))
            for cycle in cycle_indices:
                if live_monitor.stable_goal_reached() or live_monitor.goal_plane_crossed():
                    live_goal_stop_triggered = True
                    live_goal_stop_reason = "online_receding_horizon_goal_guard"
                    if first_live_goal_stop_trace is None:
                        first_live_goal_stop_trace = {
                            "decision_monotonic_ns": time.monotonic_ns(),
                            "location": "before_next_planning_cycle",
                            "next_cycle_index": cycle,
                            "held_target_raw": list(current_target),
                        }
                    break
                with live_monitor.lock:
                    px = live_monitor.last_px
                    samples = list(live_monitor.gate_samples)
                if px is None or not samples:
                    raise RuntimeError("online IPWM aborted: no reliable overhead observation")
                axis_error_px = float(np.dot(
                    online_planner.task_goal_px - np.asarray(px, dtype=float),
                    online_planner.task_axis_px,
                ))
                stagnation_observations.append({
                    "cycle": cycle,
                    "observation_monotonic_ns": int(samples[-1]["monotonic_ns"]),
                    "object_px": list(px),
                    "axis_error_px": axis_error_px,
                })
                window = args.ipwm_stagnation_cycles
                if window is not None and len(stagnation_observations) > window:
                    recent = stagnation_observations[-(window + 1):]
                    progress = float(recent[0]["axis_error_px"]) - float(recent[-1]["axis_error_px"])
                    if progress < args.ipwm_stagnation_min_progress_px:
                        live_goal_stop_triggered = True
                        live_goal_stop_reason = "online_stagnation_no_go"
                        first_live_goal_stop_trace = {
                            "decision_monotonic_ns": time.monotonic_ns(),
                            "location": "before_next_planning_cycle",
                            "next_cycle_index": cycle,
                            "held_target_raw": list(current_target),
                            "window_cycles": window,
                            "measured_axis_progress_px": progress,
                            "minimum_progress_px": args.ipwm_stagnation_min_progress_px,
                        }
                        break
                observation_ns = int(samples[-1]["monotonic_ns"])
                if observation_ns <= last_vision_ns:
                    raise RuntimeError("online IPWM aborted: stale overhead observation")
                last_vision_ns = observation_ns
                raw = np.asarray(read_positions(bus), dtype=int)
                q = ticks_to_radians(raw, zero, direction)
                now = time.monotonic()
                qd = np.zeros(5) if previous_q is None else (q - previous_q) / max(now - previous_q_time, 1e-3)
                inference_started_monotonic_ns = time.monotonic_ns()
                try:
                    plan = run_monitored_work(lambda: online_planner.plan(
                        joint_q=q, joint_qd=qd, object_px=px,
                        observation_monotonic_ns=observation_ns,
                        lock_angles=fixed_lock_angles,
                    ), lambda: monitor_until(time.monotonic() + 0.02,
                                              phase="online_inference", segment_index=cycle))
                except RuntimeError as error:
                    if str(error) != "no online IPWM candidate passes current-state safety gates":
                        raise
                    live_goal_stop_triggered = True
                    live_goal_stop_reason = "online_no_safe_candidate_no_go"
                    first_live_goal_stop_trace = {
                        "decision_monotonic_ns": time.monotonic_ns(),
                        "location": "candidate_safety_gate",
                        "next_cycle_index": cycle,
                        "held_target_raw": list(current_target),
                        "reason": str(error),
                    }
                    break
                inference_completed_monotonic_ns = time.monotonic_ns()
                target = radians_to_ticks(plan.selected_first_reference, zero, direction)
                target_tuple = tuple(int(v) for v in target)
                for index in locked_indices:
                    target_tuple = tuple(
                        first_target[j] if j == index else target_tuple[j] for j in range(5)
                    )
                cycle_path = cycle_dir / f"cycle_{cycle:02d}.npz"
                # Preserve every array without spending lock-hold time on ZIP compression.
                run_monitored_work(lambda: np.savez(
                    cycle_path,
                    observed_joint_raw=raw, observed_joint_q=q, observed_joint_qd=qd,
                    observed_object_px=np.asarray(px), scores=plan.scores,
                    selected_index=plan.selected_index,
                    selected_references=plan.selected_references,
                    selected_first_target_raw=np.asarray(target_tuple),
                    candidate_references=plan.candidates,
                    selection_eligible=plan.selection_eligible,
                    fault_mask=plan.fault_mask,
                    lock_angles=plan.lock_angles,
                    execution_reference_index=args.ipwm_online_execution_reference_index,
                ), lambda: monitor_until(time.monotonic() + 0.02,
                                          phase="online_archive", segment_index=cycle))
                cycle_record = {
                    "cycle": cycle,
                    "inference_started_monotonic_ns": inference_started_monotonic_ns,
                    "inference_completed_monotonic_ns": inference_completed_monotonic_ns,
                    "archive_completed_monotonic_ns": time.monotonic_ns(),
                    "observation_monotonic_ns": observation_ns,
                    "object_px": list(px),
                    "joint_raw": raw.tolist(),
                    "candidate_count_scored": int(len(plan.scores)),
                    "candidate_bank_sha256": plan.candidates_sha256,
                    "selected_index": plan.selected_index,
                    "selected_first_target_raw": list(target_tuple),
                    "remaining_fraction": plan.remaining_fraction,
                    "inference_s": plan.inference_s,
                    "execution_reference_index": args.ipwm_online_execution_reference_index,
                    "scoring_horizon": int(plan.candidates.shape[1]),
                    "fault_mask": plan.fault_mask.tolist(),
                    "selected_target_feedback_error_raw": (target - raw).tolist(),
                    "cycle_evidence": str(cycle_path.resolve()),
                    "cycle_evidence_sha256": sha256_file(cycle_path),
                }
                manifest["ipwm_online_replan_cycles"].append(cycle_record)
                atomic_write_json(manifest_path, manifest)
                # Expand the selected receding-horizon chunk into the same
                # bounded microstep schedule used by start alignment.  Sending
                # one distant goal would let the next camera observation occur
                # while the servo was still chasing the previous command.
                chunk_events = build_alignment_events(
                    current_target, target_tuple, safety, args.maximum_speed_deg_s
                )
                chunk_events = tuple(
                    CommandEvent(event.time_s, cycle, event.targets_raw)
                    for event in chunk_events
                )
                run_events(chunk_events, "fixed_trajectory")
                previous_q, previous_q_time = q, now
                # Require the next action to be based on a later camera frame.
                monitor_until(time.monotonic() + args.ipwm_online_observation_wait_s,
                              phase="online_observation", segment_index=cycle)
            executed_online_targets = {
                tuple(row["selected_first_target_raw"])
                for row in manifest["ipwm_online_replan_cycles"]
            }
            if (len(manifest["ipwm_online_replan_cycles"]) < 2
                    or len(executed_online_targets) < 2):
                manifest["true_closed_loop_demonstrated"] = False
                raise RuntimeError(
                    "fewer than two distinct observation-plan-action cycles; not closed loop"
                )
            manifest["true_closed_loop_demonstrated"] = True
            manifest["closed_loop_definition"] = (
                "fresh overhead observation and servo feedback -> 10k IPWM rescore -> "
                "execute first reference only -> acquire a later observation -> replan"
            )
            atomic_write_json(manifest_path, manifest)
        manifest["live_goal_early_stop"] = {
            "enabled": stop_on_live_goal,
            "triggered": live_goal_stop_triggered,
            "reason": live_goal_stop_reason,
            "first_stop_trace": first_live_goal_stop_trace,
            "stagnation_observations": stagnation_observations,
            "criterion": (
                f"{live_gate_consecutive_frames} consecutive reliable frames "
                f"within {task_goal_tolerance_px:g} px, or goal-plane crossing "
                "as a stop-only overshoot guard"
            ),
        }
        atomic_write_json(manifest_path, manifest)
        perform_goal_validation(goal_schedule.final_reason())
        monitor_until(
            time.monotonic() + args.post_roll_s,
            phase="post_roll_hold",
            segment_index=len(trajectory.waypoints) - 1,
        )

        # Normal completion also leaves no actuator energized.  This happens
        # before camera finalization so any recorder-finalization error is safe.
        if not keep_powered:
            normal_shutdown = request_and_record_shutdown("normal_completion")
            if normal_shutdown["status"] != "VERIFIED_OFF":
                raise RuntimeError(
                    "normal torque-off was not verified for every actuator: "
                    f"enabled={normal_shutdown['enabled_ids']}, "
                    f"uncertain={normal_shutdown['uncertain_ids']}"
                )
        else:
            hold_readback = tuple(
                int(bus.read_u8(servo_id, ADDRESS_TORQUE_ENABLE))
                for servo_id in SERVO_IDS
            )
            if hold_readback != (1, 1, 1, 1, 1):
                raise RuntimeError(f"powered completion hold failed: {hold_readback}")
            manifest["completion_torque_enable_readback_ids_1_5"] = list(hold_readback)
            manifest["powered_hold_active"] = True
        for recorder in recorders:
            recorder.stop()
        failures.raise_if_set()
        live_task_gate = live_monitor.gate_summary()
        live_task_gate_path = trial_dir / "live_task_gate.json"
        atomic_write_json(live_task_gate_path, live_task_gate)
        camera_validation: dict[str, dict[str, float | int]] = {}
        for recorder in recorders:
            cadence = validate_recorder_cadence(
                frame_count=recorder.frame_count,
                first_capture_mid_ns=recorder.first_capture_mid_ns,
                last_capture_mid_ns=recorder.last_capture_mid_ns,
                nominal_fps=args.video_fps,
                enforce_gate=(
                    getattr(args, "camera_cadence_policy", "strict") == "strict"
                ),
            )
            video = validate_video_file(
                recorder.output,
                expected_frames=recorder.frame_count,
                nominal_fps=args.video_fps,
            )
            camera_validation[recorder.output.name] = {**cadence, **video}
        if telemetry_count <= 0:
            raise RuntimeError("no servo telemetry rows were recorded")
        manifest.update({
            "status": "ACQUISITION_COMPLETE_UNASSESSED",
            "completed_utc": utc_now(),
            "telemetry_rows": telemetry_count,
            "command_rows": command_count,
            "microstep_dispatch_count_total": microstep_dispatch_count_total,
            "start_alignment_dispatch_count": start_alignment_dispatch_count,
            "fixed_trajectory_dispatch_count": fixed_trajectory_dispatch_count,
            "trajectory_goal_validation_count": len(goal_validation_events),
            "camera_frame_counts": {
                "daheng": daheng.frame_count,
                "directshow": directshow.frame_count,
            },
            "camera_video_validation": camera_validation,
            "live_task_gate": {
                key: value for key, value in live_task_gate.items() if key != "samples"
            },
            "artifacts": {
                "servo_telemetry": "servo_telemetry.csv",
                "commands": "commands.csv",
                "frame_timestamps": "frame_timestamps.csv",
                "daheng_video": daheng.output.name,
                "directshow_video": directshow.output.name,
                "live_task_gate": live_task_gate_path.name,
            },
            "notice": (
                "capture completed; the live gate is a recorded online diagnostic. "
                "Formal task success remains subject to offline tracking of the raw video."
            ),
        })
        atomic_write_json(manifest_path, manifest)
        completed_with_powered_hold = keep_powered
        return trial_dir
    except BaseException as error:
        # This is intentionally the first operation in the exception path.
        if bus is not None:
            request_and_record_shutdown("exception_immediate")
        # Preserve observations even when acquisition aborts; never delay shutdown.
        try:
            atomic_write_json(trial_dir / "live_task_gate.json", live_monitor.gate_summary())
        except Exception as gate_error:
            manifest["aborted_gate_save_error"] = repr(gate_error)
        manifest.update({
            "status": "ABORTED_TORQUE_OFF_REQUESTED",
            "aborted_utc": utc_now(),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "telemetry_rows": telemetry_count,
            "command_rows": command_count,
            "microstep_dispatch_count_total": microstep_dispatch_count_total,
            "start_alignment_dispatch_count": start_alignment_dispatch_count,
            "fixed_trajectory_dispatch_count": fixed_trajectory_dispatch_count,
            "trajectory_goal_validation_count": len(goal_validation_events),
            "notice": "aborted acquisition; no task outcome may be inferred",
        })
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise
    finally:
        if bus is not None and not completed_with_powered_hold:
            final_shutdown = request_and_record_shutdown("finally_repeat")
            if (final_shutdown["status"] != "VERIFIED_OFF"
                    and manifest.get("status") == "ACQUISITION_COMPLETE_UNASSESSED"):
                manifest["status"] = "ACQUISITION_COMPLETE_TORQUE_NOT_VERIFIED_OFF"
                try:
                    atomic_write_json(manifest_path, manifest)
                except Exception:
                    pass
        for recorder in started_recorders:
            recorder.stop()
        live_monitor.stop()
        timestamps.close()
        telemetry_handle.close()
        command_handle.close()
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints", type=Path, required=True,
                        help="operator-supplied CSV containing time_s and j1_raw..j5_raw")
    parser.add_argument("--trajectory-id", required=True,
                        help="operator-assigned ID; never inferred from the filename")
    parser.add_argument("--condition", choices=VALID_CONDITIONS, required=True)
    parser.add_argument(
        "--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml"
    )
    parser.add_argument(
        "--camera-settings", type=Path,
        default=ROOT / "results/real_robot/camera_settings_selected.json",
    )
    parser.add_argument("--maximum-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--maximum-lock-hold-s", type=float, default=None,
                        help="optional total lock duration cap; omitted means cycle/safety gates only")
    parser.add_argument(
        "--task-start-px", type=parse_xy_pair, default=(1086.31, 570.0),
    )
    parser.add_argument(
        "--task-goal-px", type=parse_xy_pair, default=(1116.31, 570.0),
    )
    parser.add_argument("--task-goal-tolerance-px", type=float, default=5.0)
    parser.add_argument("--task-goal-gate-mode", choices=("axiswise", "radial"), default="axiswise")
    parser.add_argument("--live-gate-consecutive-frames", type=int, default=3)
    parser.add_argument(
        "--stop-on-live-goal", action="store_true",
        help=("hold the current servo targets as soon as the live goal gate "
              "passes; the raw video and post-roll continue"),
    )
    parser.add_argument("--execute", action="store_true",
                        help="touch cameras/serial bus and execute; absent means pure dry-run")
    parser.add_argument("--acknowledge-risk", default="")
    parser.add_argument("--trial-id", default="")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/real_robot/level_a_raw")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--sdk-root", type=Path, default=Path(r"D:\GalaxySDK"))
    parser.add_argument("--telemetry-hz", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument(
        "--camera-cadence-policy", choices=("strict", "record"), default="strict",
        help=("strict aborts when measured capture cadence differs from the "
              "declared FPS; record preserves the measured cadence and warning "
              "without invalidating an otherwise complete acquisition"),
    )
    parser.add_argument("--camera-ready-timeout-s", type=float, default=8.0)
    parser.add_argument("--pre-roll-s", type=float, default=1.0)
    parser.add_argument("--post-roll-s", type=float, default=1.0)
    parser.add_argument("--maximum-start-error-deg", type=float, default=2.0)
    parser.add_argument("--minimum-voltage-v", type=float, default=6.0)
    parser.add_argument("--startup-powered-handoff", action="store_true")
    parser.add_argument("--keep-torque-enabled-after-success", action="store_true")
    parser.add_argument("--ipwm-online-archive", type=Path,
                        help="candidate NPZ; enables genuine observation-conditioned receding horizon")
    parser.add_argument("--ipwm-plane-affine", type=Path,
                        help="accepted 2-D overhead pixel-to-base affine calibration JSON")
    parser.add_argument("--ipwm-checkpoint", type=Path,
                        default=ROOT / "runs/icra_confirmation_d3_query_selective_w10/seed27/model.pt")
    parser.add_argument("--ipwm-model-config", type=Path,
                        default=ROOT / "config/experiment/icra_primary_d2d4_eval_strict_3seed_v1.yaml")
    parser.add_argument("--ipwm-axis-calibration", type=Path,
                        default=ROOT / "results/real_robot/push_axis_current_epoch_20260903.json")
    parser.add_argument("--ipwm-online-horizon", type=int, default=5)
    parser.add_argument("--ipwm-minimum-remaining-fraction", type=float, default=0.05)
    parser.add_argument("--ipwm-online-execution-reference-index", type=int, default=0,
                        help="sampled horizon index to execute after each online score")
    parser.add_argument("--ipwm-online-replans", type=int, default=None,
                        help="optional cycle cap; omitted continues until goal or stop condition")
    parser.add_argument("--ipwm-online-score-batch-size", type=int, default=10000)
    parser.add_argument("--ipwm-shared-robot-fast-path", action="store_true",
                        help="guarded exact duplicate-robot elimination; preserves IPWM object branch")
    parser.add_argument("--ipwm-online-observation-wait-s", type=float, default=0.35)
    parser.add_argument("--ipwm-stagnation-cycles", type=int,
                        help="finish as an auditable NO_GO after this many no-progress cycles")
    parser.add_argument("--ipwm-stagnation-min-progress-px", type=float, default=0.5)
    parser.add_argument("--ipwm-max-j5-excursion-deg", type=float,
                        help="optional symmetric J5 excursion bound around the frozen start")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ipwm_stagnation_cycles is not None and args.ipwm_stagnation_cycles < 2:
        raise SystemExit("--ipwm-stagnation-cycles must be at least 2")
    if args.ipwm_stagnation_min_progress_px < 0:
        raise SystemExit("--ipwm-stagnation-min-progress-px must be nonnegative")
    if (args.ipwm_max_j5_excursion_deg is not None
            and not 0 < args.ipwm_max_j5_excursion_deg <= 90):
        raise SystemExit("--ipwm-max-j5-excursion-deg must be in (0,90]")
    payload, trajectory, safety = dry_run_payload(args)
    if not args.execute:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    trial_dir = execute_hardware(args, payload, trajectory, safety)
    print(trial_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
