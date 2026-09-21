"""Read-only snapshot of STS3215 control and protection registers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_j5 import ServoBus


REGISTERS = (
    ("max_torque", 16, 2), ("p", 21, 1), ("d", 22, 1),
    ("i", 23, 1), ("startup_force", 24, 2), ("cw_deadzone", 26, 1),
    ("ccw_deadzone", 27, 1), ("protection_current", 28, 2),
    ("mode", 33, 1), ("protection_torque", 34, 1),
    ("protection_time", 35, 1), ("overload_torque", 36, 1),
    ("torque_enable", 40, 1), ("acceleration", 41, 1),
    ("goal", 42, 2), ("goal_time", 44, 2), ("goal_speed", 46, 2),
    ("torque_limit", 48, 2), ("position", 56, 2), ("load", 60, 2),
    ("voltage", 62, 1), ("temperature", 63, 1), ("status", 65, 1),
    ("moving", 66, 1), ("current", 69, 2),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bus = ServoBus(args.port)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "read_only": True,
        "servos": {},
    }
    try:
        for servo_id in range(1, 7):
            values = {}
            for name, address, size in REGISTERS:
                reader = bus.read_u8 if size == 1 else bus.read_u16
                values[name] = int(reader(servo_id, address))
            payload["servos"][str(servo_id)] = values
    finally:
        bus.close()
    payload["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
