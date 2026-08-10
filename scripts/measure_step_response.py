"""Measure synchronized step responses for J2-J5 on the connected arm."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from recover_j5 import ServoBus

ZERO = {1: 2023, 2: 2066, 3: 2058, 4: 2076, 5: 2066}
TICKS_PER_DEG = 4096 / 360
OUT = Path("hardware/calibration/raw/2026-08-10/session-03/26_step_response.csv")


def retry(fn):
    error = None
    for _ in range(4):
        try:
            return fn()
        except (TimeoutError, OSError) as exc:
            error = exc
            time.sleep(0.08)
    raise error  # type: ignore[misc]


def main() -> None:
    bus = ServoBus("COM3")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["joint", "phase", "elapsed_ms", "goal_raw", "present_raw", "error_deg", "temp_c", "current_raw"]
    try:
        for servo_id in range(1, 6):
            present = retry(lambda servo_id=servo_id: bus.read_u16(servo_id, 56))
            retry(lambda servo_id=servo_id, present=present: bus.write_u16(servo_id, 42, present))
            retry(lambda servo_id=servo_id: bus.write_u16(servo_id, 46, 0))
            retry(lambda servo_id=servo_id: bus.write_u8(servo_id, 40, 1))
        time.sleep(2.0)

        with OUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for joint in range(2, 6):
                zero = ZERO[joint]
                goal = round(zero + 10 * TICKS_PER_DEG)
                if joint == 5:
                    retry(lambda: bus.write_u8(joint, 41, 1))
                    retry(lambda: bus.write_u16(joint, 46, 0))
                else:
                    retry(lambda: bus.write_u16(joint, 46, 20))
                for servo_id in range(1, 6):
                    if servo_id != joint:
                        retry(lambda servo_id=servo_id: bus.write_u16(servo_id, 42, ZERO[servo_id]))
                start = time.perf_counter()
                retry(lambda: bus.write_u16(joint, 42, goal))
                while time.perf_counter() - start < 3.0:
                    present = retry(lambda: bus.read_u16(joint, 56))
                    temp = retry(lambda: bus.read_u8(joint, 63))
                    try:
                        current = retry(lambda: bus.read_u16(joint, 69))
                    except Exception:
                        current = "unavailable"
                    writer.writerow({"joint": f"J{joint}", "phase": "out", "elapsed_ms": round((time.perf_counter()-start)*1000, 1), "goal_raw": goal, "present_raw": present, "error_deg": round((goal-present)/TICKS_PER_DEG, 3), "temp_c": temp, "current_raw": current})
                    handle.flush()
                    time.sleep(0.05)
                retry(lambda: bus.write_u16(joint, 42, zero))
                start = time.perf_counter()
                while time.perf_counter() - start < 3.0:
                    present = retry(lambda: bus.read_u16(joint, 56))
                    temp = retry(lambda: bus.read_u8(joint, 63))
                    try:
                        current = retry(lambda: bus.read_u16(joint, 69))
                    except Exception:
                        current = "unavailable"
                    writer.writerow({"joint": f"J{joint}", "phase": "return", "elapsed_ms": round((time.perf_counter()-start)*1000, 1), "goal_raw": zero, "present_raw": present, "error_deg": round((zero-present)/TICKS_PER_DEG, 3), "temp_c": temp, "current_raw": current})
                    handle.flush()
                    time.sleep(0.05)
        print(OUT.resolve())
    finally:
        for servo_id in range(1, 6):
            try:
                present = retry(lambda servo_id=servo_id: bus.read_u16(servo_id, 56))
                retry(lambda servo_id=servo_id, present=present: bus.write_u16(servo_id, 42, present))
                retry(lambda servo_id=servo_id: bus.write_u8(servo_id, 40, 1))
            except Exception:
                pass
        bus.close()


if __name__ == "__main__":
    main()
