"""Reset the pushed cube outside a formal trial using the physical gripper.

The routine is deliberately separate from the formal-trial executor.  It opens
ID6, follows a previously audited J1-J5 trajectory to the pushed cube, closes
ID6, reverses that same trajectory, and releases the cube at the frozen start.
All phases are logged.  Any exception requests torque-off for all six axes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_j5 import ServoBus
from recover_to_frozen_start import TICKS_PER_DEG, read_retry, signed, torque_off


ACK = "I_HAVE_CLEARED_WORKSPACE_AND_CAN_CUT_POWER"


def load_path(path: Path, trajectory_id: str) -> list[tuple[float, tuple[int, ...]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["trajectory_id"] == trajectory_id]
    if len(rows) < 2:
        raise ValueError(f"trajectory {trajectory_id!r} has fewer than two waypoints")
    points = [
        (float(r["time_s"]), tuple(int(r[f"j{i}_raw"]) for i in range(1, 6)))
        for r in rows
    ]
    if points[0][0] != 0 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("trajectory time must start at zero and increase strictly")
    return points


def interpolate(points, elapsed: float) -> tuple[int, ...]:
    if elapsed <= points[0][0]:
        return points[0][1]
    if elapsed >= points[-1][0]:
        return points[-1][1]
    for (ta, qa), (tb, qb) in zip(points, points[1:]):
        if elapsed <= tb:
            alpha = (elapsed - ta) / (tb - ta)
            return tuple(round(a + alpha * (b - a)) for a, b in zip(qa, qb))
    raise AssertionError("unreachable")


def audit(points, maximum_speed_deg_s: float) -> dict:
    maximum = 0.0
    for (ta, qa), (tb, qb) in zip(points, points[1:]):
        speed = max(abs(b - a) for a, b in zip(qa, qb)) / (tb - ta) / TICKS_PER_DEG
        maximum = max(maximum, speed)
    if maximum > maximum_speed_deg_s:
        raise ValueError(f"trajectory speed {maximum:.3f} exceeds {maximum_speed_deg_s}")
    if any(not 900 <= q <= 3200 for _, qs in points for q in qs):
        raise ValueError("trajectory exceeds conservative raw joint envelope [900,3200]")
    return {"duration_s": points[-1][0], "maximum_speed_deg_s": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoints", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--maximum-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--period-s", type=float, default=0.1)
    parser.add_argument("--gripper-open-raw", type=int, default=2185)
    parser.add_argument("--gripper-close-raw", type=int, default=1050)
    parser.add_argument("--gripper-settle-s", type=float, default=2.0)
    parser.add_argument("--gripper-timeout-s", type=float, default=60.0)
    parser.add_argument("--gripper-speed-raw", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gripper-only-target-raw", type=int)
    parser.add_argument(
        "--gripper-only-allow-grasp-stall",
        action="store_true",
        help="Accept a stable, safety-bounded early stop as object contact in gripper-only mode.",
    )
    parser.add_argument("--acknowledge-risk")
    args = parser.parse_args()
    points = load_path(args.waypoints, args.trajectory_id)
    path_audit = audit(points, args.maximum_speed_deg_s)
    if not (1032 <= args.gripper_close_raw < args.gripper_open_raw <= 2203):
        raise SystemExit("gripper commands exceed measured calibration envelope")
    payload = {
        "status": "DRY_RUN_PASS",
        "formal_trial": False,
        "purpose": "out-of-trial cube reset",
        "trajectory_id": args.trajectory_id,
        "waypoints": str(args.waypoints.resolve()),
        "path_audit": path_audit,
        "gripper_open_raw": args.gripper_open_raw,
        "gripper_close_raw": args.gripper_close_raw,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "phases": [],
    }
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not args.execute:
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "reset_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print((args.output_dir / "reset_summary.json").resolve())
        return
    if args.acknowledge_risk != ACK:
        raise SystemExit(f"execution requires --acknowledge-risk {ACK}")

    bus = ServoBus(args.port)
    log_rows: list[dict] = []
    powered_hold = False
    try:
        def command_gripper(target_raw: int, *, allow_grasp_stall: bool = False) -> int:
            bus.write_u8(6, 41, 1)
            bus.write_u16(6, 46, args.gripper_speed_raw)
            bus.write_u8(6, 40, 1)
            bus.write_u16(6, 42, target_raw)
            time.sleep(0.05)
            bus.write_u16(6, 42, target_raw)
            deadline = time.monotonic() + args.gripper_timeout_s
            observed = read_retry(bus.read_u16, 6, 56)
            previous = observed
            stable_samples = 0
            while abs(observed - target_raw) > 40 and time.monotonic() < deadline:
                voltage = read_retry(bus.read_u8, 6, 62) / 10
                temperature = read_retry(bus.read_u8, 6, 63)
                current = abs(signed(read_retry(bus.read_u16, 6, 69)))
                if voltage < 6 or temperature >= 50 or current > 400:
                    raise RuntimeError(
                        f"gripper safety limit voltage={voltage} temp={temperature} current={current}"
                    )
                time.sleep(0.25)
                observed = read_retry(bus.read_u16, 6, 56)
                stable_samples = stable_samples + 1 if abs(observed - previous) <= 2 else 0
                previous = observed
                if allow_grasp_stall and observed < args.gripper_open_raw - 100 and stable_samples >= 4:
                    payload["grasp_stall_observed_raw"] = observed
                    return observed
            if abs(observed - target_raw) > 40:
                raise RuntimeError(
                    f"gripper did not reach target: target={target_raw} observed={observed}"
                )
            return observed

        modes = tuple(read_retry(bus.read_u8, i, 33) for i in range(1, 7))
        if modes != (0, 0, 0, 0, 0, 0):
            raise RuntimeError(f"all axes must be position mode, got {modes}")
        enabled = tuple(read_retry(bus.read_u8, i, 40) for i in range(1, 6))
        if enabled != (1, 1, 1, 1, 1):
            raise RuntimeError(f"J1-J5 powered handoff required, got {enabled}")
        if args.gripper_only_target_raw is not None:
            if not 1032 <= args.gripper_only_target_raw <= 2203:
                raise RuntimeError("gripper-only target exceeds measured envelope")
            observed = command_gripper(
                args.gripper_only_target_raw,
                allow_grasp_stall=args.gripper_only_allow_grasp_stall,
            )
            payload["phases"].append("gripper_only")
            payload["gripper_only_target_raw"] = args.gripper_only_target_raw
            payload["gripper_only_observed_raw"] = observed
            payload["gripper_only_allow_grasp_stall"] = (
                args.gripper_only_allow_grasp_stall
            )
            powered_hold = True
            payload["status"] = "PASS_POWERED_HOLD"
            return
        present = tuple(read_retry(bus.read_u16, i, 56) for i in range(1, 6))
        start_error = max(abs(a - b) for a, b in zip(present, points[0][1])) / TICKS_PER_DEG
        if start_error > 2.0:
            raise RuntimeError(f"start error {start_error:.3f}deg exceeds 2deg")
        command_gripper(args.gripper_open_raw)
        payload["phases"].append("open")

        def follow(path_points, phase: str) -> None:
            duration = path_points[-1][0]
            began = time.monotonic()
            steps = max(1, math.ceil(duration / args.period_s))
            for step in range(1, steps + 1):
                deadline = began + step * duration / steps
                goal = interpolate(path_points, step * duration / steps)
                for servo_id, value in enumerate(goal, 1):
                    bus.write_u16(servo_id, 42, value)
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                actual = tuple(read_retry(bus.read_u16, i, 56) for i in range(1, 7))
                voltage = min(read_retry(bus.read_u8, i, 62) / 10 for i in range(1, 7))
                temp = max(read_retry(bus.read_u8, i, 63) for i in range(1, 7))
                current = max(abs(signed(read_retry(bus.read_u16, i, 69))) for i in range(1, 7))
                if voltage < 6 or temp >= 50 or current > 400:
                    raise RuntimeError(f"safety limit voltage={voltage} temp={temp} current={current}")
                log_rows.append({"phase": phase, "elapsed_s": time.monotonic() - began,
                                 **{f"j{i}_goal_raw": goal[i-1] for i in range(1, 6)},
                                 **{f"j{i}_position_raw": actual[i-1] for i in range(1, 7)},
                                 "minimum_voltage_v": voltage, "maximum_temperature_c": temp,
                                 "maximum_abs_current_raw": current})

        follow(points, "approach")
        payload["phases"].append("approach")
        command_gripper(args.gripper_close_raw, allow_grasp_stall=True)
        payload["phases"].append("close")
        duration = points[-1][0]
        reverse = [(duration - t, q) for t, q in reversed(points)]
        follow(reverse, "return")
        payload["phases"].append("return")
        command_gripper(args.gripper_open_raw)
        payload["phases"].append("release")
        for i in range(1, 6):
            bus.write_u16(i, 42, points[0][1][i - 1])
        powered_hold = True
        payload["status"] = "PASS_POWERED_HOLD"
    except BaseException as error:
        payload["status"] = "ABORTED_TORQUE_OFF_REQUESTED"
        payload["failure_type"] = type(error).__name__
        payload["failure_message"] = str(error)
        payload["torque_off_unconfirmed_ids"] = list(torque_off(bus))
        raise
    finally:
        if not powered_hold and "torque_off_unconfirmed_ids" not in payload:
            payload["torque_off_unconfirmed_ids"] = list(torque_off(bus))
        bus.close()
        if log_rows:
            with (args.output_dir / "reset_telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
                writer.writeheader(); writer.writerows(log_rows)
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "reset_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print((args.output_dir / "reset_summary.json").resolve())


if __name__ == "__main__":
    main()
