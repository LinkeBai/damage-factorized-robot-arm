"""Slowly recover the real arm to a previously measured frozen start pose."""

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


TICKS_PER_DEG = 4096 / 360
ACK = "I_HAVE_CLEARED_WORKSPACE_AND_CAN_CUT_POWER"


def torque_off(bus: ServoBus) -> tuple[int, ...]:
    """Repeatedly request and read back torque-off for all six actuators."""
    pending = set(range(1, 7))
    for _ in range(4):
        # Sweep every still-unconfirmed actuator before doing any blocking read.
        for servo_id in sorted(pending):
            try:
                bus.write_u8(servo_id, 40, 0)
            except Exception:
                pass
        time.sleep(0.02)
        verified = set()
        for servo_id in sorted(pending):
            try:
                if read_retry(bus.read_u8, servo_id, 40) == 0:
                    verified.add(servo_id)
            except Exception:
                pass
        pending -= verified
        if not pending:
            break
        time.sleep(0.03)
    return tuple(sorted(pending))


def signed(value: int) -> int:
    return value - 65536 if value >= 32768 else value


def read_retry(read_fn, *args, attempts: int = 4):
    """Retry isolated half-duplex bus timeouts without masking a dead bus."""
    last_error = None
    for attempt in range(attempts):
        try:
            return read_fn(*args)
        except TimeoutError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.02 * (attempt + 1))
    raise TimeoutError(f"servo read failed after {attempts} attempts") from last_error


def read_confirmed_temperatures(bus, servo_ids=range(1, 6), limit_c: int = 50):
    """Reject a single corrupted bus sample without weakening overheat safety."""
    values = tuple(read_retry(bus.read_u8, i, 63) for i in servo_ids)
    if max(values) < limit_c:
        return values
    time.sleep(0.05)
    confirmed = tuple(read_retry(bus.read_u8, i, 63) for i in servo_ids)
    if max(confirmed) >= limit_c:
        return confirmed
    return confirmed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--target", default="2013,2637,2616,2734,2075")
    parser.add_argument("--speed-deg-s", type=float, default=2.0)
    parser.add_argument("--period-s", type=float, default=0.1)
    parser.add_argument("--acknowledge-risk", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settle-timeout-s", type=float, default=20.0)
    parser.add_argument("--leave-torque-enabled-on-success", action="store_true")
    parser.add_argument("--load-compensation-max-ticks", type=int, default=0)
    parser.add_argument("--maximum-final-error-deg", type=float, default=2.0)
    args = parser.parse_args()
    if args.acknowledge_risk != ACK:
        raise SystemExit(f"pass --acknowledge-risk {ACK}")
    target = tuple(int(value) for value in args.target.split(","))
    if len(target) != 5 or not 0 < args.speed_deg_s <= 2.0:
        raise SystemExit("target must have five ticks and speed must be in (0,2]")
    if not 0 < args.settle_timeout_s <= 60:
        raise SystemExit("--settle-timeout-s must be in (0,60]")
    if not 0 <= args.load_compensation_max_ticks <= 80:
        raise SystemExit("--load-compensation-max-ticks must be in [0,80]")
    if not 0 < args.maximum_final_error_deg <= 2.0:
        raise SystemExit("--maximum-final-error-deg must be in (0,2]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = args.output_dir / "telemetry.csv"
    summary_path = args.output_dir / "summary.json"
    bus = ServoBus(args.port)
    rows = []
    status = "ABORTED"
    failure = ""
    started_utc = datetime.now(timezone.utc).isoformat()
    torque_off_unconfirmed: tuple[int, ...] = tuple(range(1, 7))
    powered_hold_active = False
    torque_enable_readback: tuple[int, ...] | None = None
    try:
        start = tuple(read_retry(bus.read_u16, i, 56) for i in range(1, 6))
        modes = [read_retry(bus.read_u8, i, 33) for i in range(1, 6)]
        if modes != [0] * 5:
            raise RuntimeError(f"position modes required; got {modes}")
        for servo_id, present in enumerate(start, 1):
            bus.write_u16(servo_id, 42, present)
            bus.write_u8(servo_id, 41, 1)
            bus.write_u16(servo_id, 46, 0)
        for servo_id in range(1, 6):
            bus.write_u8(servo_id, 40, 1)
        maximum_delta = max(abs(b - a) for a, b in zip(start, target))
        duration = maximum_delta / (args.speed_deg_s * TICKS_PER_DEG)
        steps = max(1, math.ceil(duration / args.period_s))
        began = time.monotonic()
        prior_goal = start
        for step in range(1, steps + 1):
            deadline = began + step * duration / steps
            fraction = step / steps
            goal = tuple(round(a + (b - a) * fraction) for a, b in zip(start, target))
            if max(abs(a - b) for a, b in zip(prior_goal, goal)) > 3:
                raise RuntimeError("interpolated goal jump exceeded 3 ticks")
            for servo_id, value in enumerate(goal, 1):
                bus.write_u16(servo_id, 42, value)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            position = tuple(read_retry(bus.read_u16, i, 56) for i in range(1, 6))
            voltage = tuple(read_retry(bus.read_u8, i, 62) / 10 for i in range(1, 6))
            temperature = read_confirmed_temperatures(bus)
            current = tuple(signed(read_retry(bus.read_u16, i, 69)) for i in range(1, 6))
            if min(voltage) < 6.0:
                raise RuntimeError(f"undervoltage {min(voltage):.1f}V")
            if max(temperature) >= 50:
                raise RuntimeError(f"temperature {max(temperature)}C")
            if max(abs(value) for value in current) > 400:
                raise RuntimeError(f"current raw {max(abs(value) for value in current)}")
            rows.append({"step": step, "elapsed_s": time.monotonic() - began,
                         **{f"j{i}_goal_raw": goal[i-1] for i in range(1, 6)},
                         **{f"j{i}_position_raw": position[i-1] for i in range(1, 6)},
                         **{f"j{i}_current_raw": current[i-1] for i in range(1, 6)},
                         "minimum_voltage_v": min(voltage),
                         "maximum_temperature_c": max(temperature)})
            prior_goal = goal
            if step % max(1, steps // 10) == 0:
                print(f"progress={100*step/steps:.0f}% pos={position}", flush=True)
        settle_deadline = time.monotonic() + args.settle_timeout_s
        compensated_goal = list(target)
        consecutive_in_tolerance = 0
        while time.monotonic() < settle_deadline:
            for servo_id, value in enumerate(compensated_goal, 1):
                bus.write_u16(servo_id, 42, value)
            time.sleep(min(0.25, max(0.0, settle_deadline - time.monotonic())))
            final = tuple(read_retry(bus.read_u16, i, 56) for i in range(1, 6))
            voltage = tuple(read_retry(bus.read_u8, i, 62) / 10 for i in range(1, 6))
            temperature = read_confirmed_temperatures(bus)
            current = tuple(signed(read_retry(bus.read_u16, i, 69)) for i in range(1, 6))
            if min(voltage) < 6.0:
                raise RuntimeError(f"undervoltage {min(voltage):.1f}V during settle")
            if max(temperature) >= 50:
                raise RuntimeError(f"temperature {max(temperature)}C during settle")
            if max(abs(value) for value in current) > 400:
                raise RuntimeError(
                    f"current raw {max(abs(value) for value in current)} during settle"
                )
            error_deg = [abs(a - b) / TICKS_PER_DEG for a, b in zip(final, target)]
            rows.append({"step": len(rows) + 1, "elapsed_s": time.monotonic() - began,
                         **{f"j{i}_goal_raw": compensated_goal[i-1] for i in range(1, 6)},
                         **{f"j{i}_position_raw": final[i-1] for i in range(1, 6)},
                         **{f"j{i}_current_raw": current[i-1] for i in range(1, 6)},
                         "minimum_voltage_v": min(voltage),
                         "maximum_temperature_c": max(temperature)})
            consecutive_in_tolerance = (
                consecutive_in_tolerance + 1
                if max(error_deg) <= args.maximum_final_error_deg else 0
            )
            if consecutive_in_tolerance >= 3:
                break
            if args.load_compensation_max_ticks:
                for index, (actual, desired) in enumerate(zip(final, target)):
                    # A settle cycle waits at least 0.25 s before another write.
                    max_step = math.floor(args.speed_deg_s * TICKS_PER_DEG * 0.25)
                    correction = max(-max_step, min(max_step, round(0.5 * (desired - actual))))
                    if abs(desired - actual) / TICKS_PER_DEG <= args.maximum_final_error_deg:
                        correction = 0
                    uncompensated = target[index]
                    compensated_goal[index] = max(
                        uncompensated - args.load_compensation_max_ticks,
                        min(
                            uncompensated + args.load_compensation_max_ticks,
                            compensated_goal[index] + correction,
                        ),
                    )
        else:
            raise RuntimeError(f"final tracking error {max(error_deg):.3f}deg")
        if consecutive_in_tolerance < 3:
            raise RuntimeError(f"final tracking error {max(error_deg):.3f}deg")
        if args.leave_torque_enabled_on_success:
            torque_enable_readback = tuple(
                read_retry(bus.read_u8, servo_id, 40) for servo_id in range(1, 6)
            )
            if torque_enable_readback != (1, 1, 1, 1, 1):
                raise RuntimeError(
                    f"powered hold readback failed: {torque_enable_readback}"
                )
            powered_hold_active = True
        status = "PASS"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if status != "PASS" or not args.leave_torque_enabled_on_success:
            torque_off_unconfirmed = torque_off(bus)
        else:
            torque_off_unconfirmed = ()
        bus.close()
        if torque_off_unconfirmed and status == "PASS" and not powered_hold_active:
            status = "ABORTED"
            failure = (
                "torque-off readback was not confirmed for servo IDs "
                + ",".join(str(value) for value in torque_off_unconfirmed)
            )
        if rows:
            with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        summary = {"status": status, "started_utc": started_utc,
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "target_raw": target, "failure": failure,
                   "maximum_final_error_deg": args.maximum_final_error_deg,
                   "final_position_raw": final if 'final' in locals() else None,
                   "final_error_deg": error_deg if 'error_deg' in locals() else None,
                   "telemetry_rows": len(rows), "formal_trial": False,
                   "purpose": "setup recovery to frozen intact start",
                   "powered_hold_active": powered_hold_active,
                   "load_compensation_max_ticks": args.load_compensation_max_ticks,
                   "final_compensated_goal_raw": (
                       tuple(compensated_goal) if 'compensated_goal' in locals() else None
                   ),
                   "torque_enable_readback_ids_1_5": torque_enable_readback,
                   "torque_off_verified": (
                       None if powered_hold_active else not torque_off_unconfirmed
                   ),
                   "torque_off_unconfirmed_ids": torque_off_unconfirmed}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
