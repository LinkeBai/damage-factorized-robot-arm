"""Audit an archived real-IPWM candidate bank for TCP push geometry.

This is an offline kinematic audit. PASS means only that at least one candidate
respects the configured TCP pose/path gates; it does not prove contact, object
motion, or task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.prepare_real_ipwm_trial import contact_geometry_metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--axis-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-face-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-axis-alignment-error-deg", type=float, default=15.0)
    parser.add_argument("--max-height-deviation-mm", type=float, default=5.0)
    parser.add_argument("--max-lateral-deviation-mm", type=float, default=5.0)
    parser.add_argument("--max-reverse-step-mm", type=float, default=0.5)
    args = parser.parse_args()

    archive = np.load(args.candidate_archive)
    axis_payload = json.loads(args.axis_calibration.read_text(encoding="utf-8"))
    axis = np.asarray(axis_payload["diagnostics"]["unit_axis_base_xy"], dtype=float)
    metrics = contact_geometry_metrics(archive["q_reference_rad"], axis)
    gates = {
        "face": metrics["maximum_face_rotation_deg"] <= args.max_face_rotation_deg,
        "alignment": metrics["maximum_axis_alignment_error_deg"] <= args.max_axis_alignment_error_deg,
        "height": metrics["maximum_height_deviation_m"] <= args.max_height_deviation_mm / 1000.0,
        "lateral": metrics["maximum_lateral_deviation_m"] <= args.max_lateral_deviation_mm / 1000.0,
        "monotonic": metrics["maximum_reverse_step_m"] <= args.max_reverse_step_mm / 1000.0,
    }
    geometry_eligible = np.logical_and.reduce(list(gates.values()))
    preexisting_eligible = (
        archive["selection_eligible"].astype(bool)
        if "selection_eligible" in archive.files
        else np.ones(len(geometry_eligible), dtype=bool)
    )
    eligible = geometry_eligible & preexisting_eligible
    selected = int(archive["selected_candidate_index"])
    scores = archive["predicted_scores"].astype(float)
    recommended = int(np.argmin(np.where(eligible, scores, np.inf))) if np.any(eligible) else None
    selected_metrics = {name: float(values[selected]) for name, values in metrics.items()}
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if bool(np.any(eligible)) else "NO_GO",
        "candidate_count": int(len(eligible)),
        "eligible_candidate_count": int(np.sum(eligible)),
        "selected_candidate_index": selected,
        "selected_candidate_eligible": bool(eligible[selected]),
        "recommended_candidate_index": recommended,
        "recommended_predicted_score": None if recommended is None else float(scores[recommended]),
        "selected_metrics": selected_metrics,
        "gate_pass_counts": {name: int(np.sum(value)) for name, value in gates.items()},
        "preexisting_ik_speed_gate_pass_count": int(np.sum(preexisting_eligible)),
        "thresholds": {
            "maximum_face_rotation_deg": args.max_face_rotation_deg,
            "maximum_axis_alignment_error_deg": args.max_axis_alignment_error_deg,
            "maximum_height_deviation_mm": args.max_height_deviation_mm,
            "maximum_lateral_deviation_mm": args.max_lateral_deviation_mm,
            "maximum_reverse_step_mm": args.max_reverse_step_mm,
        },
        "sources": {
            "candidate_archive": {"path": str(args.candidate_archive.resolve()), "sha256": sha256(args.candidate_archive)},
            "axis_calibration": {"path": str(args.axis_calibration.resolve()), "sha256": sha256(args.axis_calibration)},
        },
        "claim_boundary": "Offline TCP path geometry only; no contact, object-displacement, closed-loop, or success claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
