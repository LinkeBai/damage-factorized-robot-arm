"""Read raw STS3215 status registers without issuing any write command."""

from __future__ import annotations

import argparse
import time

import serial


def checksum(data: bytes) -> int:
    return (~sum(data)) & 0xFF


def read_register(port: serial.Serial, servo_id: int, address: int, size: int) -> bytes:
    core = bytes((servo_id, 4, 2, address, size))
    request = b"\xff\xff" + core + bytes((checksum(core),))
    port.reset_input_buffer()
    port.write(request)
    port.flush()
    deadline = time.monotonic() + 0.2
    response = bytearray()
    while time.monotonic() < deadline:
        response.extend(port.read(port.in_waiting or 1))
        if len(response) >= 4 and len(response) >= response[3] + 4:
            return bytes(response[: response[3] + 4])
    return bytes(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    args = parser.parse_args()
    with serial.Serial(args.port, args.baudrate, timeout=0.02, write_timeout=0.1) as port:
        for servo_id in range(1, 6):
            fields = []
            for name, address, size in (("position", 56, 2), ("voltage", 62, 1), ("temperature", 63, 1)):
                packet = read_register(port, servo_id, address, size)
                valid = len(packet) >= 6 and checksum(packet[2:-1]) == packet[-1]
                error = packet[4] if len(packet) >= 5 else None
                data = packet[5:-1] if valid else b""
                fields.append(f"{name}:raw={packet.hex(' ')} valid={valid} error={error} data={data.hex(' ')}")
            print(f"ID {servo_id} | " + " | ".join(fields))


if __name__ == "__main__":
    main()
