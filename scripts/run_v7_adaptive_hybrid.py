"""V7 — Adaptive world-model-assisted IK+PD hybrid.

Improvements over V5 hybrid:
1. Adaptive correction gain: larger when WM predicts bigger deviation from reference
2. Velocity-aware damping: reduce correction when moving favorably
3. 2-step lookahead: anticipate overshoot before it happens
4. Calibration-aware: K=5 calibration should make correction more effective than K=0

Hypothesis: WM adapts to the specific residual physics of each deployment,
providing corrective signals that pure IK cannot compute.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.control_eval import infer_dfwm_context
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--latent-steps", type=int, default=30)
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    protocol = build_g1_protocol()
    split = load_target_split()
    cal = tuple(t.as_array() for t in split.calibration)
    eval_targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = collect_controller_domains(
        protocol.train, trajectories_per_domain=2, steps=100,
        seed=seed * 10_000, targets=cal
    )
    models = train_mechanism_models(
        protocol.train, train, ranges, epochs=args.epochs, device=device
    )
    print("Training complete", flush=True)

    rows = []
    for di, domain in enumerate(protocol.test):
        calibration = collect_controller_domains(
            (domain,), trajectories_per_domain=5, steps=100,
            seed=seed * 100_000 + di * 1_000, targets=cal
        )
        for shots in (0, 5):
            context = infer_dfwm_context(
                models, domain, calibration, ranges,
                shots=shots, latent_steps=args.latent_steps, device=device
            )
            wm = models.dfwm_world_model
            wm.eval()

            for ti, target in enumerate(eval_targets):
                env = MujocoArmEnv(residual_physics=domain.residual)
                locked = {
                    i: domain.damage.lock_angle_of(i)
                    for i in domain.damage.locked
                }
                locked_tuple = tuple(domain.damage.locked)

                reference, ik_error = solve_reach_reference(
                    target, env.joint_ranges, locked_joints=locked
                )
                obs = env.reset(
                    target=target, damage_config=domain.damage
                )
                reached = False
                prev_correction = np.zeros(5)

                for step in range(300):
                    # Base IK action
                    base = joint_reference_action(
                        obs["state"], reference,
                        locked_joints=locked_tuple
                    )

                    # WM prediction
                    state_t = torch.as_tensor(
                        obs["state"], dtype=torch.float32, device=device
                    ).reshape(1, -1)
                    action_t = torch.as_tensor(
                        base, dtype=torch.float32, device=device
                    ).reshape(1, -1)

                    with torch.no_grad():
                        # 2-step rollout
                        pred1, hidden = wm.step(
                            state_t, action_t, context.reshape(1, -1), None
                        )
                        # Predict where we'll be after 1 WM step
                        q1 = pred1["mean"][0, :5].cpu().numpy()

                        # Second step: what if we continue with IK?
                        pred2, _ = wm.step(
                            pred1["mean"], action_t,
                            context.reshape(1, -1), hidden
                        )
                        q2 = pred2["mean"][0, :5].cpu().numpy()

                    # Current velocity
                    qvel = obs["state"][5:]

                    # --- Adaptive correction ---
                    # 1-step position error (where WM thinks we'll be vs reference)
                    error_1step = reference - q1
                    error_2step = reference - q2

                    # Combined multi-step error (weighted toward near term)
                    position_error = 0.7 * error_1step + 0.3 * error_2step

                    # Adaptive gain: larger when prediction deviates more
                    error_mag = np.abs(position_error)
                    # Gain between 0.05 (minimal) and 0.35 (aggressive)
                    adaptive_gain = np.clip(0.08 + 1.5 * error_mag, 0.05, 0.35)

                    # Velocity damping: reduce correction if already moving toward ref
                    ref_direction = reference - obs["state"][:5]
                    ref_direction_norm = np.linalg.norm(ref_direction) + 1e-8
                    vel_alignment = np.sum(qvel * ref_direction) / ref_direction_norm
                    # Scale: -1 (moving away) -> 1.5x gain; +1 (moving toward) -> 0.5x gain
                    vel_factor = np.clip(1.0 - 0.5 * vel_alignment, 0.5, 1.5)

                    correction = adaptive_gain * vel_factor * position_error

                    # Smooth correction (don't oscillate)
                    correction = 0.7 * correction + 0.3 * prev_correction

                    # Clip to safe range
                    correction = np.clip(correction, -0.3, 0.3)
                    correction[list(domain.damage.locked)] = 0.0

                    prev_correction = correction.copy()

                    # Apply
                    action = np.clip(base + correction, -1.0, 1.0)
                    result = env.step(action)
                    obs = result["observation"]
                    distance = float(np.linalg.norm(env.ee_pos() - target))

                    if distance <= 0.05:
                        reached = True
                        break

                row = {
                    "seed": seed,
                    "domain": domain.domain_id,
                    "target": f"eval_{ti:02d}",
                    "shots": shots,
                    "success": int(reached),
                    "steps": step + 1,
                    "final_distance_m": distance,
                    "ik_error_m": ik_error,
                }
                rows.append(row)
                print(row, flush=True)

    out = Path(f"results/final/v7-adaptive-hybrid-seed{seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.resolve()}")

    # Summary
    for shots in (0, 5):
        subset = [r for r in rows if r["shots"] == shots]
        succ = sum(r["success"] for r in subset)
        avg = np.mean([r["steps"] for r in subset])
        print(f"K={shots}: {succ}/{len(subset)}, avg {avg:.1f} steps")


if __name__ == "__main__":
    main()
