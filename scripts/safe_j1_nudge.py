"""Minimal J1 relative nudge with automatic return and torque disable."""

from __future__ import annotations

import time

from recover_j5 import ServoBus


def main() -> None:
    bus = ServoBus("COM3")
    servo_id = 1
    ticks = round(2.0 * 4096 / 360)
    start = bus.read_u16(servo_id, 56)
    target = start + ticks
    print(f"J1 start={start} target={target} delta_ticks={ticks}", flush=True)
    try:
        bus.write_u16(servo_id, 42, start)
        bus.write_u8(servo_id, 41, 1)   # acceleration
        bus.write_u16(servo_id, 46, 20)  # low speed
        bus.write_u8(servo_id, 40, 1)   # torque on J1 only
        time.sleep(0.5)
        bus.write_u16(servo_id, 42, target)
        for phase in ("out", "out", "out", "out", "return", "return", "return", "return"):
            if phase == "return":
                bus.write_u16(servo_id, 42, start)
            time.sleep(0.25)
            position = bus.read_u16(servo_id, 56)
            voltage = bus.read_u8(servo_id, 62) / 10
            temperature = bus.read_u8(servo_id, 63)
            print(f"phase={phase} position={position} voltage={voltage:.1f}V temp={temperature}C", flush=True)
            if voltage < 6.0 or temperature >= 50:
                raise RuntimeError("safety threshold exceeded")
    finally:
        try:
            bus.write_u16(servo_id, 42, start)
            time.sleep(0.5)
            bus.write_u8(servo_id, 40, 0)
        finally:
            bus.close()


if __name__ == "__main__":
    main()
