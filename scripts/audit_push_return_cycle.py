"""Audit one autonomous Push-return cycle packet without hardware access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = (
    "cycle_manifest.json",
    "daheng_FDE23080341_raw.avi",
    "directshow_index1_raw.avi",
    "commands.csv",
    "servo_telemetry.csv",
    "cube_tracking.csv",
    "state_transitions.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cycle(directory: Path, maximum_motion_s: float = 10.0) -> dict:
    directory = directory.resolve()
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        errors.append("missing artifacts: " + ", ".join(missing))
    manifest_path = directory / "cycle_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid cycle_manifest.json: {exc}")

    required_true = (
        "start_gate_passed", "forward_goal_reached", "reset_goal_reached",
        "home_gate_passed", "scene_clear_gate_passed", "camera_gate_passed",
        "servo_feedback_gate_passed", "electrical_gate_passed",
    )
    for key in required_true:
        if manifest.get(key) is not True:
            errors.append(f"{key} is not true")
    duration = manifest.get("motion_duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("motion_duration_s is missing or non-numeric")
    elif not 0 < float(duration) <= maximum_motion_s:
        errors.append(
            f"motion_duration_s={duration} exceeds allowed (0,{maximum_motion_s}]"
        )
    if manifest.get("terminal_state") != "COMPLETE":
        errors.append("terminal_state is not COMPLETE")
    if manifest.get("raw_sources_immutable") is not True:
        errors.append("raw_sources_immutable is not true")

    artifacts = []
    for name in REQUIRED_FILES:
        path = directory / name
        if path.is_file():
            artifacts.append({"name": name, "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    return {
        "status": "PASS" if not errors else "FAIL",
        "cycle_directory": str(directory),
        "maximum_motion_s": maximum_motion_s,
        "motion_duration_s": duration,
        "artifacts": artifacts,
        "errors": errors,
        "claim_boundary": (
            "PASS establishes one instrumented autonomous Push-return cycle under "
            "the frozen image-space protocol; it does not establish repeated-trial "
            "reliability or unattended-operation safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cycle_directory", type=Path)
    parser.add_argument("--maximum-motion-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_cycle(args.cycle_directory, args.maximum_motion_s)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

