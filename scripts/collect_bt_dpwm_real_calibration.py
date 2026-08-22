"""Safely collect D2/D3 real-arm transitions for Z70 (dry-run by default)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from robotarm.deployment.real_calibration import (
    CalibrationPlan, build_state, radians_to_ticks, safe_excitation,
    ticks_to_radians, validate_transition_arrays)
from recover_j5 import ServoBus

ACK = "I_HAVE_CLEARED_WORKSPACE_AND_TESTED_ESTOP"


def retry(fn, attempts=4):
    error = None
    for _ in range(attempts):
        try:
            return fn()
        except (TimeoutError, OSError) as exc:
            error = exc; time.sleep(.05)
    raise error  # type: ignore[misc]


def signed_u16(value):
    return value-65536 if value >= 32768 else value


def read_vision(path: Path, maximum_age_ms: float):
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp = float(payload["timestamp_unix_s"])
    age_ms = 1000*(time.time()-timestamp)
    if age_ms < -50 or age_ms > maximum_age_ms:
        raise RuntimeError(f"stale vision pose: age={age_ms:.1f} ms")
    pose = [payload[x] for x in ("object_x_m", "object_y_m",
                                 "object_vx_m_s", "object_vy_m_s")]
    if not np.isfinite(pose).all():
        raise RuntimeError("vision pose contains non-finite values")
    return np.asarray(pose, float), age_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(
        "config/deployment/bt_dpwm_real_calibration_v1.yaml"))
    ap.add_argument("--topology", choices=("D2", "D3"), required=True)
    ap.add_argument("--budget", type=int, choices=(5, 10, 25, 50), default=50,
                    help="dry-run override only; execution must collect the full nested trajectory")
    ap.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--vision-pose-file", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/real_bt_dpwm_z70"))
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--acknowledge-risk", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collection_transitions = int(cfg.get("collection_transitions",
                                     max(cfg["transition_budgets"])))
    if args.execute and args.budget != collection_transitions:
        raise SystemExit(
            f"execution requires --budget {collection_transitions}; K budgets are nested prefixes")
    joint_map = yaml.safe_load(Path("hardware/joint_map.yaml").read_text(encoding="utf-8"))
    safety = yaml.safe_load(Path("hardware/safety_limits.yaml").read_text(encoding="utf-8"))
    locked = int(cfg["topologies"][args.topology]["locked_joint_index"])
    plan = CalibrationPlan(args.topology, locked, args.budget,
        float(cfg["frequency_hz"]), float(cfg["action_amplitude_rad"]),
        args.seed+100*args.repetition+locked)
    actions = safe_excitation(plan)
    output = args.output_dir/args.topology/f"rep{args.repetition}_k{args.budget}"
    output.mkdir(parents=True, exist_ok=True)
    plan_payload = {"config_version": cfg["version"], **plan.__dict__,
                    "execute": args.execute, "actions": actions.tolist(),
                    "budget_protocol": "nested_prefixes",
                    "inference_budgets": cfg["transition_budgets"]}
    (output/"plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    if not args.execute:
        print(f"dry-run only; wrote {output/'plan.json'}")
        return 0
    if args.acknowledge_risk != ACK:
        raise SystemExit(f"refusing motion: pass --acknowledge-risk {ACK}")
    if args.vision_pose_file is None:
        raise SystemExit("--vision-pose-file is required for execution")

    joints = joint_map["joints"]
    servo_ids = [int(x["servo_id"]) for x in joints]
    zeros = np.asarray([x["zero_raw"] for x in joints], int)
    directions = np.asarray([x["direction"] for x in joints], int)
    limits = safety["joints"]
    minimum = np.radians([x["min_deg"] for x in limits])
    maximum = np.radians([x["max_deg"] for x in limits])
    max_drift = math.radians(float(cfg["maximum_lock_drift_deg"]))
    bus, emergency = ServoBus(args.port), False
    rows, states = [], []
    try:
        raw = np.asarray([retry(lambda i=i: bus.read_u16(i, 56)) for i in servo_ids])
        q = ticks_to_radians(raw, zeros, directions)
        lock_angle = float(q[locked]); previous_q = q.copy()
        vision, vision_age = read_vision(
            args.vision_pose_file, float(cfg["maximum_vision_age_ms"]))
        states.append(build_state(q, np.zeros(5), vision))
        goals = raw.copy()
        for i, present in zip(servo_ids, raw):
            retry(lambda i=i, p=int(present): bus.write_u16(i, 42, p))
            retry(lambda i=i: bus.write_u16(i, 46, 20))
            retry(lambda i=i: bus.write_u8(i, 40, 1))
        period = 1.0/plan.frequency_hz
        for step, action in enumerate(actions):
            started = time.perf_counter()
            commanded_q = np.clip(ticks_to_radians(goals, zeros, directions)+action,
                                    minimum, maximum)
            commanded_q[locked] = lock_angle
            goals = radians_to_ticks(commanded_q, zeros, directions)
            for i, goal in zip(servo_ids, goals):
                retry(lambda i=i, goal=int(goal): bus.write_u16(i, 42, goal))
            remaining = period-(time.perf_counter()-started)
            if remaining > 0: time.sleep(remaining)
            raw = np.asarray([retry(lambda i=i: bus.read_u16(i, 56)) for i in servo_ids])
            current = np.asarray([signed_u16(retry(lambda i=i: bus.read_u16(i, 69)))
                                  for i in servo_ids])
            temperature = np.asarray([retry(lambda i=i: bus.read_u8(i, 63)) for i in servo_ids])
            q = ticks_to_radians(raw, zeros, directions); velocity = (q-previous_q)/period
            vision, vision_age = read_vision(
                args.vision_pose_file, float(cfg["maximum_vision_age_ms"]))
            if np.max(np.abs(current)) > int(cfg["abort_current_raw"]):
                raise RuntimeError("current safety threshold exceeded")
            if np.max(temperature) > float(cfg["abort_temperature_c"]):
                raise RuntimeError("temperature safety threshold exceeded")
            if abs(q[locked]-lock_angle) > max_drift:
                raise RuntimeError("locked-joint drift safety threshold exceeded")
            state = build_state(q, velocity, vision); states.append(state)
            rows.append({"step": step, "monotonic_s": time.monotonic(),
                "vision_age_ms": vision_age,
                **{f"q{j+1}_rad": q[j] for j in range(5)},
                **{f"v{j+1}_rad_s": velocity[j] for j in range(5)},
                **{f"action{j+1}": action[j] for j in range(5)},
                **{f"current{j+1}_raw": int(current[j]) for j in range(5)},
                **{f"temp{j+1}_c": int(temperature[j]) for j in range(5)},
                "object_x_m": vision[0], "object_y_m": vision[1],
                "object_vx_m_s": vision[2], "object_vy_m_s": vision[3]})
            previous_q = q
        audit = validate_transition_arrays(
            np.asarray(states), actions, locked, lock_angle, max_drift)
        np.savez_compressed(output/"transitions.npz", states=np.asarray(states),
                            actions=actions, locked_index=locked, lock_angle=lock_angle)
        with (output/"telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(output.resolve()); return 0
    except Exception:
        emergency = True
        for i in servo_ids:
            try: bus.write_u8(i, 40, 0)
            except Exception: pass
        raise
    finally:
        if not emergency:
            for i in servo_ids:
                try:
                    present = bus.read_u16(i, 56); bus.write_u16(i, 42, present)
                except Exception: pass
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
