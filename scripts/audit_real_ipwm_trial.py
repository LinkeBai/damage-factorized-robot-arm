"""Fail-closed provenance audit for a real-robot IPWM trial.

This deliberately does *not* infer model use from a trajectory name.  A trial is
an IPWM trial only when the complete decision packet exists and its hashes agree
with the files that were actually executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robotarm.deployment.fixed_raw_trajectory import VALID_CONDITIONS


REQUIRED_FILES = (
    "checkpoint",
    "model_config",
    "candidate_archive",
    "selected_trajectory",
    "action_bridge",
    "simulation_model",
    "run_manifest",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def audit(packet_path: Path) -> dict[str, Any]:
    packet_path = packet_path.resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if packet.get("schema_version") != "real_ipwm_decision_v1":
        errors.append("schema_version must be real_ipwm_decision_v1")
    if packet.get("method") != "selective_ipwm":
        errors.append("method must be selective_ipwm")
    if packet.get("planner_mode") not in {"open_loop_sequence", "receding_horizon"}:
        errors.append("planner_mode must explicitly state open_loop_sequence or receding_horizon")
    if packet.get("condition") not in VALID_CONDITIONS:
        errors.append("condition is not a canonical intact/single-lock/multi-lock label")

    files = packet.get("files")
    if not isinstance(files, dict):
        files = {}
        errors.append("files must be an object")
    resolved: dict[str, Path] = {}
    for label in REQUIRED_FILES:
        item = files.get(label)
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            errors.append(f"files.{label} requires path and sha256")
            continue
        path = _resolve(packet_path.parent, str(item["path"]))
        resolved[label] = path
        if not path.is_file():
            errors.append(f"files.{label} does not exist: {path}")
            continue
        actual = sha256(path)
        match = actual.lower() == str(item["sha256"]).lower()
        checks[f"{label}_sha256_match"] = match
        if not match:
            errors.append(f"files.{label} sha256 mismatch")

    observation = packet.get("initial_observation")
    if not isinstance(observation, dict):
        errors.append("initial_observation must be an object")
    else:
        q = np.asarray(observation.get("joint_position_rad", []), dtype=float)
        qd = np.asarray(observation.get("joint_velocity_rad_s", []), dtype=float)
        obj = np.asarray(observation.get("object_xy_m", []), dtype=float)
        goal = np.asarray(observation.get("goal_xy_m", []), dtype=float)
        checks["observation_shapes"] = [list(q.shape), list(qd.shape), list(obj.shape), list(goal.shape)]
        if q.shape != (5,) or qd.shape != (5,) or obj.shape != (2,) or goal.shape != (2,):
            errors.append("initial observation requires q[5], qd[5], object_xy_m[2], goal_xy_m[2]")
        elif not all(np.isfinite(x).all() for x in (q, qd, obj, goal)):
            errors.append("initial observation contains non-finite values")
        if not observation.get("telemetry_timestamp_utc"):
            errors.append("initial_observation.telemetry_timestamp_utc is required")
        if not (observation.get("task_plane_calibration_sha256")
                or observation.get("task_axis_calibration_sha256")):
            errors.append("an evidence-derived task calibration sha256 is required")

    decision = packet.get("decision")
    if not isinstance(decision, dict):
        decision = {}
        errors.append("decision must be an object")
    count = decision.get("candidate_count")
    selected = decision.get("selected_candidate_index")
    if not isinstance(count, int) or count < 2:
        errors.append("decision.candidate_count must be an integer >= 2")
    if not isinstance(selected, int) or not isinstance(count, int) or not 0 <= selected < count:
        errors.append("selected_candidate_index is outside candidate_count")
    if not isinstance(decision.get("predicted_scores"), list):
        errors.append("decision.predicted_scores must be archived in the packet")
    elif isinstance(count, int) and len(decision["predicted_scores"]) != count:
        errors.append("predicted_scores length does not equal candidate_count")
    if decision.get("selection_rule") != "minimum_predicted_terminal_object_to_goal_distance":
        errors.append("selection_rule is not the frozen IPWM rule")

    bridge = packet.get("bridge")
    if not isinstance(bridge, dict):
        errors.append("bridge must be an object")
    else:
        if bridge.get("simulation_model") != "genkiarm_push.xml":
            errors.append("bridge must use the calibrated genkiarm_push.xml model")
        if bridge.get("safety_audit") != "PASS":
            errors.append("bridge.safety_audit must be PASS")
        if bridge.get("clipped_action_count") not in {0, None}:
            errors.append("executed trajectory contains clipped actions")

    # Bind the decision packet to the trajectory that the hardware runner says it used.
    if "run_manifest" in resolved and "selected_trajectory" in resolved:
        try:
            run = json.loads(resolved["run_manifest"].read_text(encoding="utf-8"))
            recorded = str(run.get("trajectory_sha256") or run.get("waypoint_sha256") or "").lower()
            selected_hash = sha256(resolved["selected_trajectory"]).lower()
            checks["executed_selected_trajectory_match"] = recorded == selected_hash
            if recorded != selected_hash:
                errors.append("hardware run_manifest is not bound to selected_trajectory")
            if packet.get("trial_id") != run.get("trial_id"):
                errors.append("trial_id differs between decision packet and run_manifest")
            if packet.get("condition") != run.get("condition"):
                errors.append("condition differs between decision packet and run_manifest")
        except Exception as exc:  # fail closed on malformed evidence
            errors.append(f"could not validate run_manifest: {exc}")

    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "is_model_in_loop_evidence": not errors,
        "packet": str(packet_path),
        "checks": checks,
        "errors": errors,
        "interpretation": (
            "PASS proves archived selective-IPWM decision provenance; it does not by itself "
            "prove task success or superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.packet)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
