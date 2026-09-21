"""Pure validation and interpolation for fixed raw-tick arm trajectories.

This module deliberately has no camera, serial-port, or robot imports.  It is
safe to import in tests and is the only code path used by the runner's default
dry-run mode.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5")
RAW_COLUMNS = tuple(f"{name}_raw" for name in JOINT_NAMES)
SINGLE_LOCK_CONDITIONS = {f"D{i}": i - 1 for i in range(1, 6)}
MULTI_LOCK_CONDITIONS = tuple(
    "+".join(f"J{i + 1}" for i in indices)
    for count in range(2, 6)
    for indices in itertools.combinations(range(5), count)
)
VALID_CONDITIONS = ("intact", *SINGLE_LOCK_CONDITIONS, *MULTI_LOCK_CONDITIONS)
LOCK_INDEX_BY_CONDITION = dict(SINGLE_LOCK_CONDITIONS)


def locked_indices_for_condition(condition: str) -> tuple[int, ...]:
    """Return canonical zero-based locked indices for all 31 fault subsets."""
    if condition == "intact":
        return ()
    if condition in SINGLE_LOCK_CONDITIONS:
        return (SINGLE_LOCK_CONDITIONS[condition],)
    if condition not in MULTI_LOCK_CONDITIONS:
        raise ValueError(f"condition must be one of {VALID_CONDITIONS}")
    return tuple(int(token[1:]) - 1 for token in condition.split("+"))
TICKS_PER_DEGREE = 4096.0 / 360.0


@dataclass(frozen=True)
class JointSafety:
    name: str
    servo_id: int
    zero_raw: int
    direction: int
    min_deg: float
    max_deg: float
    max_speed_deg_s: float

    @property
    def min_raw(self) -> int:
        endpoints = (
            self.zero_raw + self.direction * self.min_deg * TICKS_PER_DEGREE,
            self.zero_raw + self.direction * self.max_deg * TICKS_PER_DEGREE,
        )
        return math.ceil(min(endpoints) - 1e-12)

    @property
    def max_raw(self) -> int:
        endpoints = (
            self.zero_raw + self.direction * self.min_deg * TICKS_PER_DEGREE,
            self.zero_raw + self.direction * self.max_deg * TICKS_PER_DEGREE,
        )
        return math.floor(max(endpoints) + 1e-12)

    @property
    def max_speed_ticks_s(self) -> float:
        return self.max_speed_deg_s * TICKS_PER_DEGREE


@dataclass(frozen=True)
class SafetyEnvelope:
    joints: tuple[JointSafety, ...]
    abort_current_raw: int
    abort_temp_c: float
    max_lock_drift_deg: float


@dataclass(frozen=True)
class RawWaypoint:
    waypoint_index: int
    time_s: float
    targets_raw: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class FixedRawTrajectory:
    trajectory_id: str
    condition: str
    source: Path
    source_sha256: str
    waypoints: tuple[RawWaypoint, ...]
    maximum_commanded_speed_deg_s: float

    @property
    def duration_s(self) -> float:
        return self.waypoints[-1].time_s

    @property
    def locked_joint_index(self) -> int | None:
        indices = self.locked_joint_indices
        return indices[0] if len(indices) == 1 else None

    @property
    def locked_joint_indices(self) -> tuple[int, ...]:
        return locked_indices_for_condition(self.condition)

    @property
    def locked_joint_name(self) -> str | None:
        index = self.locked_joint_index
        return None if index is None else JOINT_NAMES[index]


@dataclass(frozen=True)
class CommandEvent:
    """One integer target on the original trajectory time base."""

    time_s: float
    segment_index: int
    targets_raw: tuple[int, int, int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safety_envelope(path: Path) -> SafetyEnvelope:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("joints") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(JOINT_NAMES):
        raise ValueError("safety file must define exactly five positioning joints")
    joints: list[JointSafety] = []
    for expected_name, row in zip(JOINT_NAMES, rows):
        if not isinstance(row, dict) or row.get("name") != expected_name:
            raise ValueError(f"safety joint order must be {JOINT_NAMES}")
        required = (
            "servo_id", "zero_raw", "direction", "min_deg", "max_deg",
            "max_speed_deg_s",
        )
        if any(row.get(key) is None for key in required):
            raise ValueError(f"{expected_name}: measured safety fields are incomplete")
        joint = JointSafety(
            name=expected_name,
            servo_id=int(row["servo_id"]),
            zero_raw=int(row["zero_raw"]),
            direction=int(row["direction"]),
            min_deg=float(row["min_deg"]),
            max_deg=float(row["max_deg"]),
            max_speed_deg_s=float(row["max_speed_deg_s"]),
        )
        if joint.direction not in (-1, 1):
            raise ValueError(f"{expected_name}: direction must be -1 or 1")
        if joint.max_speed_deg_s <= 0.0:
            raise ValueError(f"{expected_name}: max speed must be positive")
        joints.append(joint)
    if [joint.servo_id for joint in joints] != [1, 2, 3, 4, 5]:
        raise ValueError("this executor requires canonical servo IDs 1..5")
    damage = payload.get("damage_test", {})
    for key in ("abort_current_raw", "abort_temp_c", "max_lock_drift_deg"):
        if damage.get(key) is None:
            raise ValueError(f"safety file damage_test.{key} is required")
    return SafetyEnvelope(
        joints=tuple(joints),
        abort_current_raw=int(damage["abort_current_raw"]),
        abort_temp_c=float(damage["abort_temp_c"]),
        max_lock_drift_deg=float(damage["max_lock_drift_deg"]),
    )


def _parse_integer(value: str, label: str) -> int:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} is blank")
    try:
        parsed = int(stripped, 10)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer raw tick value") from exc
    return parsed


def _selected_rows(
    rows: Sequence[Mapping[str, str]], trajectory_id: str, condition: str,
    has_trajectory_column: bool, has_condition_column: bool,
) -> list[Mapping[str, str]]:
    selected: list[Mapping[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if has_trajectory_column and row.get("trajectory_id", "").strip() != trajectory_id:
            continue
        if has_condition_column and row.get("condition", "").strip() != condition:
            if has_trajectory_column:
                if row.get("trajectory_id", "").strip() == trajectory_id:
                    raise ValueError(
                        f"row {row_number}: trajectory {trajectory_id!r} condition does not "
                        f"match requested {condition!r}"
                    )
                continue
            raise ValueError(
                f"row {row_number}: condition does not match requested {condition!r}"
            )
        selected.append(row)
    return selected


def load_fixed_raw_trajectory(
    path: Path,
    *,
    trajectory_id: str,
    condition: str,
    safety: SafetyEnvelope,
    maximum_speed_deg_s: float = 5.0,
) -> FixedRawTrajectory:
    """Load one user-supplied raw-tick trajectory and reject unsafe ambiguity."""
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {VALID_CONDITIONS}")
    if not trajectory_id.strip():
        raise ValueError("trajectory_id must be non-empty and supplied by the operator")
    if not math.isfinite(maximum_speed_deg_s) or maximum_speed_deg_s <= 0.0:
        raise ValueError("maximum_speed_deg_s must be finite and positive")
    configured_maximum = min(joint.max_speed_deg_s for joint in safety.joints)
    if maximum_speed_deg_s > configured_maximum + 1e-12:
        raise ValueError(
            f"requested speed {maximum_speed_deg_s:g} deg/s exceeds configured "
            f"{configured_maximum:g} deg/s"
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"time_s", *RAW_COLUMNS}
        if missing := required - fields:
            raise ValueError(f"waypoint CSV missing columns: {sorted(missing)}")
        radian_columns = set(JOINT_NAMES)
        if fields & radian_columns and not radian_columns <= fields:
            raise ValueError("waypoint CSV must provide either all or none of j1..j5 radians")
        rows = list(reader)
    selected = _selected_rows(
        rows,
        trajectory_id,
        condition,
        "trajectory_id" in fields,
        "condition" in fields,
    )
    if not selected:
        qualifier = "matching rows" if "trajectory_id" in fields else "waypoint rows"
        raise ValueError(f"waypoint CSV contains no {qualifier} for {trajectory_id!r}")
    parsed: list[RawWaypoint] = []
    for offset, row in enumerate(selected):
        row_number = offset + 2
        try:
            timestamp = float(row["time_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number}: time_s must be numeric") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"row {row_number}: time_s must be finite")
        index = (
            _parse_integer(row.get("waypoint_index", ""), f"row {row_number} waypoint_index")
            if "waypoint_index" in fields
            else offset
        )
        targets = tuple(
            _parse_integer(row[column], f"row {row_number} {column}")
            for column in RAW_COLUMNS
        )
        if set(JOINT_NAMES) <= fields:
            for target, joint in zip(targets, safety.joints):
                try:
                    supplied_rad = float(row[joint.name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"row {row_number} {joint.name} radians must be numeric"
                    ) from exc
                expected_rad = math.radians(
                    joint.direction * (target - joint.zero_raw) / TICKS_PER_DEGREE
                )
                if not math.isfinite(supplied_rad) or abs(supplied_rad - expected_rad) > 1e-8:
                    raise ValueError(
                        f"row {row_number}: {joint.name}/{joint.name}_raw disagree"
                    )
        parsed.append(RawWaypoint(index, timestamp, targets))  # type: ignore[arg-type]
    parsed.sort(key=lambda waypoint: waypoint.waypoint_index)
    if len(parsed) < 2:
        raise ValueError("trajectory requires at least two waypoints")
    if [item.waypoint_index for item in parsed] != list(range(len(parsed))):
        raise ValueError("waypoint_index must start at 0 and be contiguous")
    if abs(parsed[0].time_s) > 1e-12:
        raise ValueError("first waypoint time_s must be 0")
    for previous, current in zip(parsed, parsed[1:]):
        if current.time_s <= previous.time_s:
            raise ValueError("waypoint time_s values must be strictly increasing")
    for waypoint in parsed:
        for target, joint in zip(waypoint.targets_raw, safety.joints):
            if target < joint.min_raw or target > joint.max_raw:
                raise ValueError(
                    f"waypoint {waypoint.waypoint_index}: {joint.name} target {target} "
                    f"is outside measured raw range [{joint.min_raw}, {joint.max_raw}]"
                )
    for locked_index in locked_indices_for_condition(condition):
        locked_values = {item.targets_raw[locked_index] for item in parsed}
        if len(locked_values) != 1:
            raise ValueError(
                f"{condition} requires locked {JOINT_NAMES[locked_index]} target to remain "
                "exactly constant for every waypoint"
            )
    maximum_observed = 0.0
    for previous, current in zip(parsed, parsed[1:]):
        dt = current.time_s - previous.time_s
        for index, joint in enumerate(safety.joints):
            speed = (
                abs(current.targets_raw[index] - previous.targets_raw[index])
                / TICKS_PER_DEGREE
                / dt
            )
            maximum_observed = max(maximum_observed, speed)
            allowed = min(maximum_speed_deg_s, joint.max_speed_deg_s)
            if speed > allowed + 1e-9:
                raise ValueError(
                    f"segment {previous.waypoint_index}->{current.waypoint_index}: "
                    f"{joint.name} speed {speed:.6g} deg/s exceeds {allowed:g} deg/s"
                )
    return FixedRawTrajectory(
        trajectory_id=trajectory_id,
        condition=condition,
        source=path.resolve(),
        source_sha256=sha256_file(path),
        waypoints=tuple(parsed),
        maximum_commanded_speed_deg_s=maximum_observed,
    )


def _integer_progress(delta: int, step: int, steps: int) -> int:
    """Return a monotone rounded fraction whose increments are at most one tick."""
    magnitude = (abs(delta) * step + steps // 2) // steps
    return magnitude if delta >= 0 else -magnitude


def interpolate_raw_waypoints(trajectory: FixedRawTrajectory) -> tuple[CommandEvent, ...]:
    """Expand waypoints into <=1-tick events without changing their time base.

    Using one event for each tick of the largest-moving joint makes the discrete
    command-rate bound equal to the already-audited endpoint rate; quantization
    can therefore never create a hidden command jump above 5 deg/s.
    """
    first = trajectory.waypoints[0]
    events = [CommandEvent(first.time_s, first.waypoint_index, first.targets_raw)]
    for segment_index, (start, stop) in enumerate(
        zip(trajectory.waypoints, trajectory.waypoints[1:])
    ):
        delta = tuple(b - a for a, b in zip(start.targets_raw, stop.targets_raw))
        steps = max(abs(value) for value in delta)
        if steps == 0:
            events.append(CommandEvent(stop.time_s, segment_index, stop.targets_raw))
            continue
        duration = stop.time_s - start.time_s
        for step in range(1, steps + 1):
            fraction_time = start.time_s + duration * step / steps
            target = tuple(
                origin + _integer_progress(change, step, steps)
                for origin, change in zip(start.targets_raw, delta)
            )
            events.append(CommandEvent(fraction_time, segment_index, target))  # type: ignore[arg-type]
    return tuple(events)


def validate_interpolated_events(
    events: Iterable[CommandEvent], safety: SafetyEnvelope,
    maximum_speed_deg_s: float,
) -> float:
    """Re-audit the actual integer event sequence and return its max speed."""
    rows = tuple(events)
    if not rows:
        raise ValueError("interpolated command sequence is empty")
    maximum = 0.0
    for previous, current in zip(rows, rows[1:]):
        dt = current.time_s - previous.time_s
        if dt <= 0.0:
            if current.targets_raw != previous.targets_raw:
                raise ValueError("interpolated target changed at a non-increasing time")
            continue
        for index, joint in enumerate(safety.joints):
            speed = (
                abs(current.targets_raw[index] - previous.targets_raw[index])
                / TICKS_PER_DEGREE
                / dt
            )
            maximum = max(maximum, speed)
            allowed = min(maximum_speed_deg_s, joint.max_speed_deg_s)
            if speed > allowed + 1e-8:
                raise ValueError(
                    f"interpolated {joint.name} event speed {speed:.6g} deg/s "
                    f"exceeds {allowed:g} deg/s"
                )
    return maximum


def minimum_safe_command_interval_s(
    previous: Sequence[int], current: Sequence[int], safety: SafetyEnvelope,
    maximum_speed_deg_s: float,
) -> float:
    """Minimum wall-clock interval required between two integer commands."""
    if len(previous) != len(JOINT_NAMES) or len(current) != len(JOINT_NAMES):
        raise ValueError("commands must contain five raw targets")
    return max(
        abs(after - before)
        / (min(maximum_speed_deg_s, joint.max_speed_deg_s) * TICKS_PER_DEGREE)
        for before, after, joint in zip(previous, current, safety.joints)
    )


def build_alignment_events(
    present_raw: Sequence[int], first_target_raw: Sequence[int],
    safety: SafetyEnvelope, maximum_speed_deg_s: float,
) -> tuple[CommandEvent, ...]:
    """Build a bounded, explicitly labelled alignment from feedback to waypoint 0."""
    start = tuple(int(value) for value in present_raw)
    target = tuple(int(value) for value in first_target_raw)
    if len(start) != len(JOINT_NAMES) or len(target) != len(JOINT_NAMES):
        raise ValueError("alignment requires five present and five target ticks")
    delta = tuple(after - before for before, after in zip(start, target))
    steps = max(abs(value) for value in delta)
    if steps == 0:
        return ()
    duration = max(
        abs(change)
        / (min(maximum_speed_deg_s, joint.max_speed_deg_s) * TICKS_PER_DEGREE)
        for change, joint in zip(delta, safety.joints)
    )
    events: list[CommandEvent] = []
    for step in range(1, steps + 1):
        event_target = tuple(
            origin + _integer_progress(change, step, steps)
            for origin, change in zip(start, delta)
        )
        events.append(CommandEvent(duration * step / steps, -1, event_target))  # type: ignore[arg-type]
    return tuple(events)


def validate_camera_settings(payload: Mapping[str, object]) -> dict[str, float]:
    required = (
        "daheng_exposure", "daheng_gain", "second_exposure", "second_brightness",
    )
    output: dict[str, float] = {}
    for key in required:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"camera settings require numeric {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"camera setting {key} must be finite")
        output[key] = value
    return output
