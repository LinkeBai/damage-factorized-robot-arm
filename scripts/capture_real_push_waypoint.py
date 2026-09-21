"""Read J1-J5 once and append an operator-labelled raw/radian waypoint.

This teaching helper is read-only with respect to the servo bus: it never
writes a register, enables torque, or moves an actuator.  The operator must
provide the condition, trajectory ID, waypoint index, and time explicitly.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    JOINT_NAMES,
    TICKS_PER_DEGREE,
    VALID_CONDITIONS,
    SafetyEnvelope,
    load_safety_envelope,
)


SERVO_IDS = (1, 2, 3, 4, 5)
ADDRESS_PRESENT_POSITION = 56
FIELDS = [
    "trajectory_id", "condition", "waypoint_index", "time_s", "captured_utc",
    *[f"{name}_raw" for name in JOINT_NAMES],
    *JOINT_NAMES,
]


def raw_to_radians(raw: Sequence[int], safety: SafetyEnvelope) -> tuple[float, ...]:
    if len(raw) != len(JOINT_NAMES):
        raise ValueError("five raw positions are required")
    return tuple(
        math.radians(joint.direction * (value - joint.zero_raw) / TICKS_PER_DEGREE)
        for value, joint in zip(raw, safety.joints)
    )


def capture_positions_readonly(bus: object) -> tuple[int, int, int, int, int]:
    """Issue exactly five present-position reads and no bus writes."""
    return tuple(
        int(bus.read_u16(servo_id, ADDRESS_PRESENT_POSITION))
        for servo_id in SERVO_IDS
    )  # type: ignore[return-value]


def append_waypoint(
    output: Path,
    *,
    trajectory_id: str,
    condition: str,
    waypoint_index: int,
    time_s: float,
    raw: Sequence[int],
    safety: SafetyEnvelope,
    captured_utc: str,
) -> None:
    if not trajectory_id.strip():
        raise ValueError("trajectory_id must be non-empty")
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {VALID_CONDITIONS}")
    if waypoint_index < 0:
        raise ValueError("waypoint_index must be nonnegative")
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_s must be finite and nonnegative")
    values = tuple(int(item) for item in raw)
    if len(values) != len(JOINT_NAMES):
        raise ValueError("five raw positions are required")
    for value, joint in zip(values, safety.joints):
        if value < joint.min_raw or value > joint.max_raw:
            raise ValueError(
                f"{joint.name} feedback {value} outside measured range "
                f"[{joint.min_raw}, {joint.max_raw}]"
            )
    existing: list[dict[str, str]] = []
    write_header = not output.exists() or output.stat().st_size == 0
    if output.exists() and output.stat().st_size > 0:
        with output.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or ()) != FIELDS:
                raise ValueError(
                    "existing waypoint CSV schema differs; refusing an ambiguous append"
                )
            existing = list(reader)
    same = [row for row in existing
            if row["trajectory_id"] == trajectory_id and row["condition"] == condition]
    conflicting = [row for row in existing
                   if row["trajectory_id"] == trajectory_id
                   and row["condition"] != condition]
    if conflicting:
        raise ValueError("one trajectory_id cannot be reused across conditions")
    if same:
        indices = [int(row["waypoint_index"]) for row in same]
        times = [float(row["time_s"]) for row in same]
        expected_index = max(indices) + 1
        if waypoint_index != expected_index:
            raise ValueError(
                f"next waypoint_index for {trajectory_id!r}/{condition} must be "
                f"{expected_index}"
            )
        if time_s <= max(times):
            raise ValueError("new time_s must be strictly later than the prior waypoint")
    elif waypoint_index != 0 or abs(time_s) > 1e-12:
        raise ValueError("the first waypoint must use waypoint_index=0 and time_s=0")
    radians = raw_to_radians(values, safety)
    row: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "condition": condition,
        "waypoint_index": waypoint_index,
        "time_s": f"{time_s:.9f}",
        "captured_utc": captured_utc,
    }
    row.update({f"{name}_raw": value for name, value in zip(JOINT_NAMES, values)})
    row.update({name: f"{value:.12f}" for name, value in zip(JOINT_NAMES, radians)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--condition", choices=VALID_CONDITIONS, required=True)
    parser.add_argument("--waypoint-index", type=int, required=True)
    parser.add_argument("--time-s", type=float, required=True)
    parser.add_argument("--port", default="COM3")
    parser.add_argument(
        "--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    safety = load_safety_envelope(args.safety)
    from scripts.recover_j5 import ServoBus

    bus = ServoBus(args.port)
    try:
        raw = capture_positions_readonly(bus)
    finally:
        bus.close()
    append_waypoint(
        args.output,
        trajectory_id=args.trajectory_id,
        condition=args.condition,
        waypoint_index=args.waypoint_index,
        time_s=args.time_s,
        raw=raw,
        safety=safety,
        captured_utc=datetime.now(timezone.utc).isoformat(),
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
