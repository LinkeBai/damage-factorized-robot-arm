"""Run the 10-minute G0 electrical lock-repeatability check."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from recover_j5 import ServoBus

ZEROS = [2023, 2066, 2058, 2076, 2066]
TICKS_PER_DEG = 4096 / 360


def retry(operation, attempts: int = 4):
    last_error = None
    for _ in range(attempts):
        try:
            return operation()
        except (TimeoutError, OSError) as error:
            last_error = error
            time.sleep(0.12)
    raise last_error  # type: ignore[misc]


def read_u8(bus: ServoBus, servo_id: int, address: int) -> int:
    return retry(lambda: bus.read_u8(servo_id, address))


def read_u16(bus: ServoBus, servo_id: int, address: int) -> int:
    return retry(lambda: bus.read_u16(servo_id, address))


def write_u8(bus: ServoBus, servo_id: int, address: int, value: int) -> None:
    retry(lambda: bus.write_u8(servo_id, address, value))


def write_u16(bus: ServoBus, servo_id: int, address: int, value: int) -> None:
    retry(lambda: bus.write_u16(servo_id, address, value))


def move(bus: ServoBus, target_deg: list[float], duration: float = 2.0) -> None:
    start = [read_u16(bus, i, 56) for i in range(1, 6)]
    target = [round(z + d * TICKS_PER_DEG) for z, d in zip(ZEROS, target_deg)]
    steps = max(1, round(duration / 0.1))
    for step in range(1, steps + 1):
        for servo_id, (origin, goal) in enumerate(zip(start, target), 1):
            write_u16(bus, servo_id, 42, round(origin + (goal - origin) * step / steps))
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--max-drift", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    bus = ServoBus(args.port)
    max_drift = 0.0
    max_temp = 0
    cycles = 0
    try:
        for servo_id in range(1, 6):
            write_u8(bus, servo_id, 41, 1)
            write_u16(bus, servo_id, 46, 0)
            write_u8(bus, servo_id, 40, 1)
        move(bus, [0, 0, 0, 0, 0], 2.0)
        time.sleep(3.0)
        started = time.monotonic()

        with args.out.open("w", newline="", encoding="ascii") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_s", "cycle", "locked_joint", "phase", "locked_deg", "drift_deg", "max_temp_c"])
            while time.monotonic() - started < args.duration:
                locked = 2 + cycles % 3
                baseline = read_u16(bus, locked, 56)
                # Exercise a neighboring pitch joint while preserving the lock goal.
                pose = [0.0] * 5
                moving = 2 if locked != 2 else 3
                pose[moving - 1] = 10 if cycles % 2 == 0 else -10
                move(bus, pose, 2.0)
                time.sleep(2.0)
                present = read_u16(bus, locked, 56)
                drift = abs(present - baseline) / TICKS_PER_DEG
                temps = [read_u8(bus, i, 63) for i in range(1, 6)]
                max_drift = max(max_drift, drift)
                max_temp = max(max_temp, *temps)
                elapsed = time.monotonic() - started
                writer.writerow([f"{elapsed:.1f}", cycles + 1, f"J{locked}", "loaded_hold", f"{(present-ZEROS[locked-1])/TICKS_PER_DEG:.2f}", f"{drift:.2f}", max(temps)])
                handle.flush()
                if drift > args.max_drift or max(temps) >= 55:
                    raise RuntimeError(f"safety threshold exceeded: drift={drift:.2f} deg, temp={max(temps)} C")
                move(bus, [0, 0, 0, 0, 0], 2.0)
                time.sleep(3.0)
                cycles += 1
                print(f"cycle={cycles} elapsed={elapsed:.0f}s lock=J{locked} drift={drift:.2f}deg temp_max={max(temps)}C", flush=True)
        print(f"PASS cycles={cycles} max_drift={max_drift:.2f}deg max_temp={max_temp}C", flush=True)
    except Exception:
        for servo_id in range(1, 6):
            try:
                present = read_u16(bus, servo_id, 56)
                write_u16(bus, servo_id, 42, present)
                write_u8(bus, servo_id, 40, 1)
            except Exception:
                pass
        raise
    finally:
        bus.close()


if __name__ == "__main__":
    main()
