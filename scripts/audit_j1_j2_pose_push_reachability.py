"""Audit pose-constrained Push reachability for the frozen J1+J2 fault.

This is an offline kinematic boundary audit.  It does not access hardware and
does not relabel any real trial.  J1, J2 and J5 are held at the frozen start;
J3/J4 are exhaustively swept inside joint and 10 s at 5 deg/s travel limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from robotarm.deployment.fixed_raw_trajectory import load_safety_envelope
from robotarm.deployment.real_calibration import ticks_to_radians
from scripts.prepare_real_ipwm_trial import forward_kinematics_batched


FROZEN_RAW = np.asarray([2085, 2635, 2603, 2740, 2077])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=1000)
    parser.add_argument("--height-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--lateral-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grid < 100:
        raise SystemExit("--grid must be at least 100")

    safety_path = ROOT / "hardware/safety_limits.yaml"
    axis_path = ROOT / "results/real_robot/push_axis_current_epoch_20260903.json"
    safety = load_safety_envelope(safety_path)
    axis_doc = json.loads(axis_path.read_text(encoding="utf-8"))
    zero = np.asarray([j.zero_raw for j in safety.joints])
    direction = np.asarray([j.direction for j in safety.joints])
    q0 = ticks_to_radians(FROZEN_RAW, zero, direction)

    # Frozen protocol: 5 deg/s for at most 10 s permits at most 50 degrees of
    # travel from the start on each free axis.
    travel = np.deg2rad(50.0)
    j3 = np.linspace(max(np.deg2rad(-92), q0[2] - travel),
                     min(np.deg2rad(92), q0[2] + travel), args.grid)
    j4 = np.linspace(max(np.deg2rad(-94), q0[3] - travel),
                     min(np.deg2rad(92), q0[3] + travel), args.grid)
    q3, q4 = np.meshgrid(j3, j4, indexing="ij")
    candidates = np.repeat(q0[None, :], q3.size, axis=0)
    candidates[:, 2] = q3.ravel()
    candidates[:, 3] = q4.ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = []
    for start in range(0, len(candidates), 100_000):
        batch = torch.as_tensor(candidates[start:start + 100_000],
                                dtype=torch.float64, device=device)
        positions.append(forward_kinematics_batched(batch).cpu().numpy())
    positions = np.concatenate(positions)
    start_position = forward_kinematics_batched(
        torch.as_tensor(q0[None, :], dtype=torch.float64, device=device)
    ).cpu().numpy()[0]

    axis = np.asarray(axis_doc["diagnostics"]["unit_axis_base_xy"], dtype=float)
    normal = np.asarray([-axis[1], axis[0]])
    delta_xy = positions[:, :2] - start_position[:2]
    projected = delta_xy @ axis
    lateral = delta_xy @ normal
    # With J1/J2/J5 fixed, the push-face pitch change is exactly the change in
    # q3+q4.  Keeping J5 fixed also preserves push-face yaw.
    face_error_deg = np.abs(np.degrees(
        candidates[:, 2] + candidates[:, 3] - q0[2] - q0[3]
    ))
    base_gate = (
        (np.abs(positions[:, 2] - start_position[2]) <= args.height_tolerance_mm / 1000)
        & (np.abs(lateral) <= args.lateral_tolerance_mm / 1000)
    )
    metres_per_pixel = float(axis_doc["diagnostics"]["metres_per_pixel"])
    rows = []

    # Exact zero-rotation manifold: with J1/J2/J5 frozen, preserving the TCP
    # orientation requires q3 + q4 to remain constant.  Sweep this one degree
    # of freedom directly rather than approximating zero with a 2-D grid.
    exact_j3 = np.linspace(
        max(np.deg2rad(-92), q0[2] - travel),
        min(np.deg2rad(92), q0[2] + travel),
        max(args.grid * args.grid, 100_000),
    )
    exact_j4 = q0[2] + q0[3] - exact_j3
    exact_limit = (
        (exact_j4 >= max(np.deg2rad(-94), q0[3] - travel))
        & (exact_j4 <= min(np.deg2rad(92), q0[3] + travel))
    )
    exact_candidates = np.repeat(q0[None, :], len(exact_j3), axis=0)
    exact_candidates[:, 2] = exact_j3
    exact_candidates[:, 3] = exact_j4
    exact_positions = []
    for start in range(0, len(exact_candidates), 100_000):
        batch = torch.as_tensor(
            exact_candidates[start:start + 100_000], dtype=torch.float64, device=device
        )
        exact_positions.append(forward_kinematics_batched(batch).cpu().numpy())
    exact_positions = np.concatenate(exact_positions)
    exact_delta_xy = exact_positions[:, :2] - start_position[:2]
    exact_projected = exact_delta_xy @ axis
    exact_lateral = exact_delta_xy @ normal
    exact_gate = (
        exact_limit
        & (np.abs(exact_positions[:, 2] - start_position[2]) <= args.height_tolerance_mm / 1000)
        & (np.abs(exact_lateral) <= args.lateral_tolerance_mm / 1000)
    )
    exact_best = int(np.argmax(np.where(exact_gate, exact_projected, -np.inf)))
    exact_zero_rotation = {
        "constraint": "exact_constant_tcp_orientation_q3_plus_q4_constant_j1_j2_j5_fixed",
        "candidate_count": int(len(exact_candidates)),
        "feasible_candidate_count": int(np.sum(exact_gate)),
        "maximum_projected_tcp_travel_m": float(exact_projected[exact_best]),
        "maximum_projected_tcp_travel_equivalent_px": float(
            exact_projected[exact_best] / metres_per_pixel
        ),
        "best_joint_deg": np.degrees(exact_candidates[exact_best]).tolist(),
        "best_height_error_mm": float(
            1000 * (exact_positions[exact_best, 2] - start_position[2])
        ),
        "best_lateral_error_mm": float(1000 * exact_lateral[exact_best]),
        "orientation_change_deg": 0.0,
    }
    for threshold in (5, 10, 15, 20, 25, 30):
        gate = base_gate & (face_error_deg <= threshold)
        best = int(np.argmax(np.where(gate, projected, -np.inf)))
        rows.append({
            "maximum_face_error_deg": threshold,
            "feasible_candidate_count": int(np.sum(gate)),
            # This is TCP kinematic travel, not measured or simulated object
            # displacement.  Contact effectiveness is outside this audit.
            "maximum_projected_tcp_travel_m": float(projected[best]),
            "maximum_projected_tcp_travel_equivalent_px": float(
                projected[best] / metres_per_pixel
            ),
            "best_joint_deg": np.degrees(candidates[best]).tolist(),
            "best_height_error_mm": float(1000 * (positions[best, 2] - start_position[2])),
            "best_lateral_error_mm": float(1000 * lateral[best]),
            "best_face_error_deg": float(face_error_deg[best]),
        })

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_accessed": False,
        "condition": "J1+J2",
        "frozen_start_raw": FROZEN_RAW.tolist(),
        "grid_per_axis": args.grid,
        "candidate_count": int(len(candidates)),
        "device": str(device),
        "frozen_speed_deg_s": 5.0,
        "frozen_lock_hold_s": 10.0,
        "height_tolerance_mm": args.height_tolerance_mm,
        "lateral_tolerance_mm": args.lateral_tolerance_mm,
        "target_object_displacement_px": 30.0,
        "exact_zero_rotation": exact_zero_rotation,
        "rows": rows,
        "source_hashes": {
            "safety": digest(safety_path),
            "axis_calibration": digest(axis_path),
            "analytic_fk": digest(ROOT / "src/robotarm/envs/fk.py"),
        },
        "conclusion": (
            "The exact constant-orientation manifold and bounded orientation-error "
            "sweeps report TCP translation only. They do not establish a required "
            "rotation for object displacement, because contact dynamics are absent."
        ),
        "claim_boundary": (
            "Dense offline TCP kinematic boundary audit; it does not model "
            "contact, object displacement, task success, or statistical generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
