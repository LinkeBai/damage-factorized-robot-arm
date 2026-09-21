"""Audit every interpolated event of one selected raw trajectory in GenkiArm MuJoCo."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotarm.deployment.fixed_raw_trajectory import (  # noqa: E402
    TICKS_PER_DEGREE, interpolate_raw_waypoints, load_fixed_raw_trajectory,
    load_safety_envelope, validate_interpolated_events,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "sim/assets/genkiarm_push.xml")
    parser.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    safety = load_safety_envelope(args.safety)
    trajectory = load_fixed_raw_trajectory(
        args.trajectory, trajectory_id=args.trajectory_id, condition=args.condition,
        safety=safety, maximum_speed_deg_s=5.0,
    )
    events = interpolate_raw_waypoints(trajectory)
    maximum_speed = validate_interpolated_events(events, safety, 5.0)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    forbidden = []
    intentional_block_contacts = 0
    for event_index, event in enumerate(events):
        q = [
            math.radians(joint.direction * (raw - joint.zero_raw) / TICKS_PER_DEGREE)
            for raw, joint in zip(event.targets_raw, safety.joints)
        ]
        data.qpos[:5] = np.asarray(q)
        mujoco.mj_forward(model, data)
        for contact in data.contact[:data.ncon]:
            first = model.geom(int(contact.geom1)).name
            second = model.geom(int(contact.geom2)).name
            if "block_geom" in (first, second):
                intentional_block_contacts += 1
                continue
            forbidden.append({
                "event_index": event_index, "time_s": event.time_s,
                "geom1": first, "geom2": second, "distance_m": float(contact.dist),
            })
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_accessed": False,
        "condition": args.condition,
        "trajectory_id": args.trajectory_id,
        "trajectory_sha256": sha256(args.trajectory),
        "model_sha256": sha256(args.model),
        "safety_sha256": sha256(args.safety),
        "interpolated_event_count": len(events),
        "maximum_interpolated_speed_deg_s": maximum_speed,
        "forbidden_collision_count": len(forbidden),
        "intentional_block_contact_sample_count": intentional_block_contacts,
        "collision_gate": "PASS" if not forbidden else "FAIL",
        "forbidden_collisions": forbidden,
        "claim_boundary": (
            "Collision-proxy audit of the selected open-loop reference only; provisional "
            "MuJoCo contact geometry is not a substitute for real clearance validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "condition", "interpolated_event_count", "maximum_interpolated_speed_deg_s",
        "forbidden_collision_count", "intentional_block_contact_sample_count", "collision_gate",
    )}, indent=2))
    return 0 if not forbidden else 2


if __name__ == "__main__":
    raise SystemExit(main())
