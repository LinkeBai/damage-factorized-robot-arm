"""Audited all-axis STS3215 position-P update with fail-closed shutdown."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_j5 import ServoBus
from recover_to_frozen_start import torque_off


ACK = "I_AM_PHYSICALLY_SUPPORTING_THE_ARM_AND_CAN_CUT_POWER"


def snapshot(bus: ServoBus) -> dict[str, dict[str, int]]:
    return {
        str(servo_id): {
            "p": int(bus.read_u8(servo_id, 21)),
            "mode": int(bus.read_u8(servo_id, 33)),
            "torque_enable": int(bus.read_u8(servo_id, 40)),
            "goal_raw": int(bus.read_u16(servo_id, 42)),
            "position_raw": int(bus.read_u16(servo_id, 56)),
            "torque_limit": int(bus.read_u16(servo_id, 48)),
            "voltage_raw": int(bus.read_u8(servo_id, 62)),
            "temperature_c": int(bus.read_u8(servo_id, 63)),
            "status": int(bus.read_u8(servo_id, 65)),
        }
        for servo_id in range(1, 7)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--p-gain", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-supported-arm", required=True)
    args = parser.parse_args()
    if args.acknowledge_supported_arm != ACK:
        raise SystemExit(f"pass --acknowledge-supported-arm {ACK}")
    if not 5 <= args.p_gain <= 32:
        raise SystemExit("--p-gain must be in conservative range [5,32]")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_p_gain": args.p_gain,
        "status": "IN_PROGRESS",
        "support_acknowledgement": ACK,
    }
    bus = ServoBus(args.port)
    try:
        payload["before"] = snapshot(bus)
        if any(v["mode"] != 0 for k, v in payload["before"].items() if k != "6"):
            raise RuntimeError("J1-J5 must all be in position mode 0")
        failed_off = torque_off(bus)
        payload["pre_write_torque_off_unconfirmed_ids"] = list(failed_off)
        if failed_off:
            raise RuntimeError(f"torque-off not verified for IDs {failed_off}")
        # Address 55 unlocks EEPROM writes. Keep every actuator off throughout
        # the entire write/readback phase.
        for servo_id in range(1, 6):
            bus.write_u8(servo_id, 55, 0)
            bus.write_u8(servo_id, 21, args.p_gain)
            time.sleep(0.02)
            observed = int(bus.read_u8(servo_id, 21))
            if observed != args.p_gain:
                raise RuntimeError(
                    f"P-gain readback mismatch ID {servo_id}: {observed}"
                )
            bus.write_u8(servo_id, 55, 1)
        payload["after_write_torque_off"] = snapshot(bus)
        # This tool deliberately does not re-enable torque. Recovery must seed
        # safe present-position goals before enabling the five joints.
        payload["status"] = "PASS_TORQUE_REMAINS_OFF"
    except BaseException as error:
        payload["status"] = "ABORTED_TORQUE_OFF_REQUESTED"
        payload["failure_type"] = type(error).__name__
        payload["failure_message"] = str(error)
        payload["exception_torque_off_unconfirmed_ids"] = list(torque_off(bus))
        raise
    finally:
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        bus.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
