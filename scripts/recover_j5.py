"""Recover servo ID5 from a latched velocity-mode command.

Run this while the arm power is off, then turn the arm power on. The script
never enables torque. It restores position mode and seeds the goal with the
measured position before reporting success.
"""

from __future__ import annotations

import argparse
import time

import serial


HEADER = bytes((0xFF, 0xFF))
INST_READ = 2
INST_WRITE = 3


def checksum(data: bytes) -> int:
    return (~sum(data)) & 0xFF


class ServoBus:
    READ_DEADLINE_S = 0.12

    def __init__(self, port: str, baudrate: int = 1_000_000) -> None:
        self.serial = serial.Serial(port, baudrate, timeout=0.08, write_timeout=0.08)

    def close(self) -> None:
        self.serial.close()

    def send(
        self, servo_id: int, instruction: int, params: bytes, *, flush: bool = True
    ) -> None:
        core = bytes((servo_id, len(params) + 2, instruction)) + params
        self.serial.write(HEADER + core + bytes((checksum(core),)))
        if flush:
            self.serial.flush()

    def write_u8(self, servo_id: int, address: int, value: int) -> None:
        self.send(servo_id, INST_WRITE, bytes((address, value & 0xFF)))

    def write_u16(self, servo_id: int, address: int, value: int) -> None:
        self.send(
            servo_id,
            INST_WRITE,
            bytes((address, value & 0xFF, (value >> 8) & 0xFF)),
        )

    def write_u16_batch(self, writes) -> None:
        """Send the same ordered write packets with one serial flush."""
        packets = []
        for servo_id, address, value in writes:
            if not (1 <= servo_id <= 253 and 0 <= address <= 254 and 0 <= value <= 65535):
                raise ValueError("invalid batched register write")
            params = bytes((address, value & 0xFF, (value >> 8) & 0xFF))
            core = bytes((servo_id, len(params) + 2, INST_WRITE)) + params
            packets.append(HEADER + core + bytes((checksum(core),)))
        if packets:
            payload = b"".join(packets)
            if self.serial.write(payload) != len(payload):
                raise TimeoutError("incomplete batched serial write")
            self.serial.flush()

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        deadline = time.monotonic() + self.READ_DEADLINE_S
        self.serial.reset_input_buffer()
        # ``write`` is already bounded by pyserial's write_timeout.  Avoid an
        # additional unbounded flush so this request and response share one
        # aggregate deadline.
        self.send(servo_id, INST_READ, bytes((address, size)), flush=False)
        packet = bytearray()
        original_timeout = self.serial.timeout
        try:
            while True:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    break
                # A fixed per-byte timeout can overrun the aggregate deadline
                # on the final read.  Bound every blocking read by the actual
                # remaining budget so READ_DEADLINE_S is a total deadline.
                self.serial.timeout = min(float(original_timeout), remaining_s)
                byte = self.serial.read(1)
                if not byte:
                    continue
                packet += byte
                while len(packet) >= 2 and packet[:2] != HEADER:
                    del packet[0]
                if len(packet) >= 4:
                    total = packet[3] + 4
                    if len(packet) >= total:
                        response = bytes(packet[:total])
                        if checksum(response[2:-1]) != response[-1]:
                            del packet[:total]
                            continue
                        if response[2] != servo_id:
                            del packet[:total]
                            continue
                        if response[4] != 0:
                            raise RuntimeError(f"servo error byte {response[4]}")
                        data = response[5:-1]
                        if len(data) != size:
                            del packet[:total]
                            continue
                        return data
        finally:
            self.serial.timeout = original_timeout
        raise TimeoutError("servo did not respond")

    def read_u8(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[0]

    def read_u16(self, servo_id: int, address: int) -> int:
        data = self.read(servo_id, address, 2)
        return data[0] | (data[1] << 8)


def recover(port: str, timeout_s: float) -> int:
    servo_id = 5
    bus = ServoBus(port)
    deadline = time.monotonic() + timeout_s
    print("armed: keep clear, then power on the arm")
    try:
        while time.monotonic() < deadline:
            # These writes are harmless while unpowered and stop a latched motor
            # as soon as the bus becomes live.
            bus.write_u8(servo_id, 40, 0)  # torque off
            bus.write_u16(servo_id, 46, 0)  # velocity command zero
            try:
                present = bus.read_u16(servo_id, 56)
            except TimeoutError:
                time.sleep(0.03)
                continue

            bus.write_u8(servo_id, 40, 0)
            bus.write_u16(servo_id, 46, 0)
            bus.write_u8(servo_id, 55, 0)  # unlock EEPROM
            bus.write_u8(servo_id, 33, 0)  # position mode
            time.sleep(0.08)
            bus.write_u16(servo_id, 42, present)  # hold current pose if enabled later
            bus.write_u8(servo_id, 40, 0)
            bus.write_u8(servo_id, 55, 1)
            # Model 777 uses a nonzero speed word as a fixed direction hint.
            # Zero lets position mode choose the direction from the target.
            bus.write_u16(servo_id, 46, 0)
            time.sleep(0.1)

            mode = bus.read_u8(servo_id, 33)
            torque = bus.read_u8(servo_id, 40)
            speed = bus.read_u16(servo_id, 46)
            goal = bus.read_u16(servo_id, 42)
            present = bus.read_u16(servo_id, 56)
            print(
                f"recovered: mode={mode} torque={torque} speed={speed} "
                f"goal={goal} present={present}"
            )
            if mode == 0 and torque == 0 and speed == 0:
                return 0
            raise RuntimeError("recovery verification failed")
        print("timeout: no powered response from ID5")
        return 2
    finally:
        try:
            bus.write_u8(servo_id, 40, 0)
        finally:
            bus.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    return recover(args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
