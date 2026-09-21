"""Bind an IPWM preparation packet to its immutable real execution evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_real_ipwm_trial import audit, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    trial = args.trial_dir.resolve()
    run = trial / "run_manifest.json"
    tracking = trial / "offline_cube_tracking_v1/yellow_cube_summary.json"
    for path in (run, tracking):
        if not path.is_file():
            raise SystemExit(f"missing required evidence: {path}")
    tracking_payload = json.loads(tracking.read_text(encoding="utf-8"))
    run_payload = json.loads(run.read_text(encoding="utf-8"))
    packet = dict(preparation)
    packet["schema_version"] = "real_ipwm_decision_v1"
    # A preparation packet may live outside the immutable execution directory.
    # Keep both identifiers explicit, while making ``trial_id`` denote the
    # physical run audited below.  The trajectory SHA-256 remains the binding
    # between the decision and execution.
    packet["decision_trial_id"] = packet.get("trial_id")
    packet["trial_id"] = run_payload.get("trial_id")
    if not packet.get("initial_observation", {}).get("telemetry_timestamp_utc"):
        from datetime import datetime, timezone
        packet["initial_observation"]["telemetry_timestamp_utc"] = datetime.fromtimestamp(
            args.preparation.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        packet["initial_observation"]["timestamp_source"] = (
            "preparation_manifest_filesystem_mtime_posthoc; subsecond telemetry binding unavailable"
        )
    packet["files"] = dict(packet["files"])
    # The preparation source may have gained provenance-only fields after a
    # run.  Bind the final packet to the auditable source and exact MJCF now;
    # the numerical candidates/scores remain immutable in candidate_archive.
    prepare_source = Path(__file__).with_name("prepare_real_ipwm_trial.py").resolve()
    simulation_model = Path(__file__).resolve().parents[1] / "sim/assets/genkiarm_push.xml"
    packet["files"]["action_bridge"] = {
        "path": str(prepare_source), "sha256": sha256(prepare_source),
    }
    packet["files"]["simulation_model"] = {
        "path": str(simulation_model.resolve()), "sha256": sha256(simulation_model),
    }
    packet["files"]["run_manifest"] = {"path": str(run), "sha256": sha256(run)}
    packet["files"]["tracking_summary"] = {"path": str(tracking), "sha256": sha256(tracking)}
    endpoint_xy = tracking_payload["image_task"]["endpoint_error_xy_px"]
    axis_success = abs(float(endpoint_xy[0])) <= 5.0 and abs(float(endpoint_xy[1])) <= 5.0
    packet["outcome"] = {
        "success": axis_success,
        "endpoint_error_xy_px": tracking_payload["image_task"]["endpoint_error_xy_px"],
        "endpoint_error_radial_px": tracking_payload["image_task"]["endpoint_error_radial_px"],
        "displacement_px": tracking_payload["displacement_px"],
        "criterion": "axis-wise |dx|<=5 px and |dy|<=5 px",
    }
    output = trial / "model_decision_manifest.json"
    output.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    result = audit(output)
    audit_path = trial / "model_decision_audit.json"
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "success": packet["outcome"]["success"],
                      "manifest": str(output), "audit": str(audit_path),
                      "errors": result["errors"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
