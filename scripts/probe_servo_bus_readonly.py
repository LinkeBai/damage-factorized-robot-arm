"""Read-only servo-bus probe; never writes registers or enables torque."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_j5 import ServoBus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--servo-ids", default="1,2,3,4,5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = [int(item) for item in args.servo_ids.split(",")]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "baudrate": args.baudrate,
        "read_only": True,
        "registers": {"position_raw": 56, "temperature_c": 63},
        "servos": [],
        "status": "FAIL",
    }
    bus = None
    try:
        bus = ServoBus(args.port, args.baudrate)
        for servo_id in ids:
            item = {"servo_id": servo_id}
            try:
                item["position_raw"] = bus.read_u16(servo_id, 56)
                item["temperature_c"] = bus.read_u8(servo_id, 63)
                item["responded"] = True
            except Exception as error:
                item["responded"] = False
                item["error"] = str(error)
            payload["servos"].append(item)
        payload["status"] = "PASS" if all(x["responded"] for x in payload["servos"]) else "FAIL"
    except Exception as error:
        payload["error"] = str(error)
    finally:
        if bus is not None:
            bus.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["status"])
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
