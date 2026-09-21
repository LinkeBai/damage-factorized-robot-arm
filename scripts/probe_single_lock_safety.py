"""Fail-closed, low-amplitude, non-contact lock probe for J1/J5 authorization.

The default mode is an offline plan only.  ``--execute`` is required to open the
servo bus.  The probe holds every arm joint under power, freezes one joint at
its measured start position, perturbs one other joint by a small angle, returns
to the measured start, and records position/current/temperature/voltage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from recover_j5 import ServoBus
from robotarm.deployment.fixed_raw_trajectory import (
    VALID_CONDITIONS,
    locked_indices_for_condition,
)

ROOT = Path(__file__).resolve().parents[1]
TICKS_PER_DEG = 4096.0 / 360.0
FROZEN_PUSH_START_RAW = (2085, 2635, 2603, 2740, 2077)


def signed_u16(value: int) -> int:
    return value - 65536 if value >= 32768 else value


def retry(operation, attempts: int = 4):
    last = None
    for _ in range(attempts):
        try:
            return operation()
        except (TimeoutError, OSError) as exc:
            last = exc
            time.sleep(0.1)
    raise last  # type: ignore[misc]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", choices=("j1", "j5"))
    parser.add_argument(
        "--condition",
        choices=tuple(value for value in VALID_CONDITIONS if value != "intact"),
        help="Canonical single- or multi-lock condition; mutually exclusive with --joint",
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--amplitude-deg", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-pose-raw", default=",".join(map(str, FROZEN_PUSH_START_RAW)))
    parser.add_argument("--expected-pose-tolerance-deg", type=float, default=12.0)
    args = parser.parse_args()
    try:
        expected_pose = tuple(int(value) for value in args.expected_pose_raw.split(","))
    except ValueError:
        raise SystemExit("expected pose requires five integer raw positions")
    if len(expected_pose) != 5 or any(not 0 <= value <= 4095 for value in expected_pose):
        raise SystemExit("expected pose requires five raw positions in [0, 4095]")
    if not 0 < args.expected_pose_tolerance_deg <= 12:
        raise SystemExit("expected pose tolerance must be in (0, 12] degrees")

    if (args.joint is None) == (args.condition is None):
        raise SystemExit("pass exactly one of --joint or --condition")

    if not (0.25 <= args.amplitude_deg <= 3.0):
        raise SystemExit("amplitude must be in [0.25, 3.0] deg")
    # Fifteen seconds is the longest duration already exercised by the loaded
    # lock-validation protocol. Longer holds require a separate authorization.
    if not (3.0 <= args.hold_s <= 15.0):
        raise SystemExit("hold time must be in [3, 15] s")

    config_path = ROOT / "hardware" / "safety_limits.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    condition = args.condition or f"D{int(args.joint[1:])}"
    locked_ids = tuple(index + 1 for index in locked_indices_for_condition(condition))
    joint_id = locked_ids[0]
    # Prefer a useful pitch/distal perturbation, but never command a locked axis.
    moving_id = next((servo_id for servo_id in (2, 4, 3, 5, 1)
                      if servo_id not in locked_ids), None)
    if moving_id is None:
        raise SystemExit("all five axes are locked; no non-contact perturbation axis exists")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.joint or condition.replace("+", "_").lower()
    out = args.out or ROOT / "results" / "real_robot" / f"lock_probe_{label}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "manifest.json"
    telemetry_path = out / "telemetry.csv"
    manifest = {
        "schema": "single-lock-safety-probe-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRY_RUN" if not args.execute else "RUNNING",
        "joint": args.joint,
        "condition": condition,
        "joint_id": joint_id,
        "locked_joint_ids": list(locked_ids),
        "locked_joints": [f"j{servo_id}" for servo_id in locked_ids],
        "moving_joint_id": moving_id,
        "amplitude_deg": args.amplitude_deg,
        "hold_s": args.hold_s,
        "all_arm_joints_powered": bool(args.execute),
        "non_contact_protocol": True,
        "expected_pose_raw": list(expected_pose),
        "expected_pose_tolerance_deg": args.expected_pose_tolerance_deg,
        "script_sha256": sha256(Path(__file__)),
        "safety_config": str(config_path.relative_to(ROOT)),
        "safety_config_sha256": sha256(config_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        print(out.resolve())
        return

    limits = cfg["damage_test"]
    abort_current = int(limits["abort_current_raw"])
    abort_temp = int(limits["abort_temp_c"])
    max_drift_deg = float(limits["max_lock_drift_deg"])
    min_voltage_v = float(limits["min_bus_voltage_v"])
    max_voltage_v = float(limits["max_bus_voltage_v"])
    joint_cfg = {int(j["servo_id"]): j for j in cfg["joints"]}
    bus = ServoBus(args.port)
    rows: list[list[object]] = []
    status = "ABORT"
    reason = "unknown"
    start: list[int] = []
    lock_start: dict[int, int] = {}
    max_current = 0
    max_temp = 0
    max_drift = 0.0
    max_drift_by_joint = {servo_id: 0.0 for servo_id in locked_ids}

    def read_all(elapsed: float, phase: str) -> None:
        nonlocal max_current, max_temp, max_drift
        positions = [retry(lambda i=i: bus.read_u16(i, 56)) for i in range(1, 6)]
        currents = [signed_u16(retry(lambda i=i: bus.read_u16(i, 69))) for i in range(1, 6)]
        temps = [retry(lambda i=i: bus.read_u8(i, 63)) for i in range(1, 6)]
        voltage = retry(lambda: bus.read_u8(1, 62)) / 10.0
        drifts = {
            servo_id: abs(positions[servo_id - 1] - lock_start[servo_id]) / TICKS_PER_DEG
            for servo_id in locked_ids
        }
        drift = max(drifts.values())
        for servo_id, value in drifts.items():
            max_drift_by_joint[servo_id] = max(max_drift_by_joint[servo_id], value)
        max_current = max(max_current, *(abs(v) for v in currents))
        max_temp = max(max_temp, *temps)
        max_drift = max(max_drift, drift)
        rows.append([
            f"{elapsed:.3f}", phase, *positions, *currents, *temps, voltage,
            f"{drift:.4f}",
            *[f"{drifts.get(i, 0.0):.4f}" for i in range(1, 6)],
        ])
        if not min_voltage_v <= voltage <= max_voltage_v:
            raise RuntimeError(
                f"bus voltage outside [{min_voltage_v:.1f}, {max_voltage_v:.1f}] V: "
                f"{voltage:.1f} V"
            )
        if max(abs(v) for v in currents) > abort_current:
            raise RuntimeError(f"overcurrent raw: {max(abs(v) for v in currents)}")
        if max(temps) >= abort_temp:
            raise RuntimeError(f"overtemperature: {max(temps)} C")
        if drift > max_drift_deg:
            raise RuntimeError(f"lock drift: {drift:.3f} deg")

    try:
        start = [retry(lambda i=i: bus.read_u16(i, 56)) for i in range(1, 6)]
        lock_start = {servo_id: start[servo_id - 1] for servo_id in locked_ids}
        # The non-contact probe is performed at the frozen Push operating pose,
        # not at the calibration zeros.  Fail closed if the current pose has
        # drifted materially from that auditable experiment baseline.
        for i, (observed, expected) in enumerate(
            zip(start, expected_pose), 1
        ):
            error_deg = abs(observed - expected) / TICKS_PER_DEG
            if error_deg > args.expected_pose_tolerance_deg:
                raise RuntimeError(
                    f"J{i} is not near frozen Push start "
                    f"(observed={observed}, expected={expected}, "
                    f"error_deg={error_deg:.3f}); no motion issued"
                )
        target = start.copy()
        target[moving_id - 1] += round(args.amplitude_deg * TICKS_PER_DEG)
        min_raw = int(joint_cfg[moving_id]["zero_raw"] + joint_cfg[moving_id]["min_deg"] * TICKS_PER_DEG)
        max_raw = int(joint_cfg[moving_id]["zero_raw"] + joint_cfg[moving_id]["max_deg"] * TICKS_PER_DEG)
        if not min_raw <= target[moving_id - 1] <= max_raw:
            raise RuntimeError("probe target violates joint limit")

        # Seed every goal first, then enable every joint.  The locked goal never changes.
        for i, present in enumerate(start, 1):
            retry(lambda i=i, present=present: bus.write_u16(i, 42, present))
            retry(lambda i=i: bus.write_u8(i, 41, 1))
            retry(lambda i=i: bus.write_u16(i, 46, 40))
        for i in range(1, 6):
            retry(lambda i=i: bus.write_u8(i, 40, 1))

        begun = time.monotonic()
        read_all(0.0, "powered_settle")
        time.sleep(1.0)
        retry(lambda: bus.write_u16(moving_id, 42, target[moving_id - 1]))
        while (elapsed := time.monotonic() - begun) < 1.0 + args.hold_s:
            read_all(elapsed, "loaded_hold")
            time.sleep(0.1)
        retry(lambda: bus.write_u16(moving_id, 42, start[moving_id - 1]))
        for _ in range(10):
            read_all(time.monotonic() - begun, "return")
            time.sleep(0.1)
        status, reason = "PASS", "all gates passed"
    except Exception as exc:
        reason = str(exc)
        # Preserve powered support against gravity when communication still works:
        # seed all current positions, keeping every axis under closed-loop hold.
        for i in range(1, 6):
            try:
                present = retry(lambda i=i: bus.read_u16(i, 56))
                retry(lambda i=i, present=present: bus.write_u16(i, 42, present))
                retry(lambda i=i: bus.write_u8(i, 40, 1))
            except Exception:
                pass
    finally:
        bus.close()
        with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_s", "phase", *[f"j{i}_position_raw" for i in range(1, 6)],
                             *[f"j{i}_current_raw" for i in range(1, 6)],
                             *[f"j{i}_temperature_c" for i in range(1, 6)], "voltage_v", "lock_drift_deg",
                             *[f"j{i}_lock_drift_deg" for i in range(1, 6)]])
            writer.writerows(rows)
        manifest.update({"status": status, "reason": reason, "start_raw": start,
                         "max_current_raw": max_current, "max_temperature_c": max_temp,
                         "max_lock_drift_deg": max_drift,
                         "max_lock_drift_by_joint_deg": {
                             f"j{servo_id}": value
                             for servo_id, value in max_drift_by_joint.items()
                         },
                         "telemetry_sha256": sha256(telemetry_path)})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(out.resolve())
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
