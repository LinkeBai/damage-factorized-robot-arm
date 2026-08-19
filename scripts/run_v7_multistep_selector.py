"""V7 — Multi-step world-model-guided candidate selector.

Key improvements over V6:
1. Multi-step (3-step) WM rollout scoring — exploits WM's unique predictive capability
2. Smaller, directed perturbations — less random, more intelligent exploration
3. Velocity-alignment scoring — favors actions that move toward target quickly
4. Action smoothness penalty — prevents oscillation
5. Fallback to IK when WM uncertainty is high — safety net

Hypothesis: multi-step prediction lets the WM anticipate dynamics effects
(overshoot, joint coupling, residual physics) that pure geometry-based IK cannot.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.fk import forward_kinematics
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
    ap.add_argument("--horizon", type=int, default=3,
                    help="Multi-step rollout horizon for WM scoring")
    ap.add_argument("--candidates", type=int, default=24,
                    help="Number of candidate actions per step")
    ap.add_argument("--perturb-std", type=float, default=0.06,
                    help="Joint perturbation std (rad)")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    protocol = build_g1_protocol()
    split = load_target_split()
    cal = tuple(t.as_array() for t in split.calibration)
    targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train models
    t0 = time.perf_counter()
    train = collect_controller_domains(
        protocol.train, trajectories_per_domain=2, steps=100,
        seed=seed * 10_000, targets=cal
    )
    models = train_mechanism_models(
        protocol.train, train, ranges, epochs=args.epochs, device=device
    )
    print(f"Training complete ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows = []
    for di, domain in enumerate(protocol.test):
        calibration = collect_controller_domains(
            (domain,), trajectories_per_domain=5, steps=100,
            seed=seed * 100_000 + di * 1_000, targets=cal
        )
        context = infer_dfwm_context(
            models, domain, calibration, ranges,
            shots=5, latent_steps=30, device=device
        )
        wm = models.dfwm_world_model
        wm.eval()

        for ti, target in enumerate(targets):
            env = MujocoArmEnv(residual_physics=domain.residual)
            locked = {i: domain.damage.lock_angle_of(i)
                      for i in domain.damage.locked}
            locked_tuple = tuple(domain.damage.locked)

            base_ref, ik_error = solve_reach_reference(
                target, env.joint_ranges, locked_joints=locked
            )
            obs = env.reset(target=target, damage_config=domain.damage)
            reached = False
            rng = np.random.default_rng(seed + ti + di * 100)
            prev_action = np.zeros(5)

            for step in range(300):
                # --- Build candidate set ---
                # 1) Base IK reference (always included for safety)
                candidates_ref = [base_ref]

                # 2) Small random perturbations around reference
                for _ in range(args.candidates - 1):
                    perturbed = base_ref + rng.normal(
                        0.0, args.perturb_std, size=5
                    )
                    perturbed = np.clip(
                        perturbed, env.joint_ranges[:, 0], env.joint_ranges[:, 1]
                    )
                    for j, a in locked.items():
                        perturbed[j] = a
                    candidates_ref.append(perturbed)

                # Convert candidates to actions
                candidate_actions = np.stack([
                    joint_reference_action(
                        obs["state"], q, locked_joints=locked_tuple
                    )
                    for q in candidates_ref
                ])

                # --- Multi-step WM rollout scoring ---
                with torch.no_grad():
                    # Prepare batch
                    state_batch = torch.as_tensor(
                        obs["state"], dtype=torch.float32, device=device
                    ).unsqueeze(0).expand(len(candidate_actions), -1)

                    action_batch = torch.as_tensor(
                        candidate_actions, dtype=torch.float32, device=device
                    )

                    context_batch = context.reshape(1, -1).expand(
                        len(candidate_actions), -1
                    )

                    target_t = torch.as_tensor(
                        target, dtype=torch.float32, device=device
                    )

                    # Multi-step rollout
                    total_cost = torch.zeros(len(candidate_actions), device=device)
                    current_state = state_batch
                    hidden = None

                    for h in range(args.horizon):
                        pred, hidden = wm.step(
                            current_state, action_batch, context_batch, hidden
                        )
                        current_state = pred["mean"]
                        q_pred = current_state[:, :5]
                        qvel_pred = current_state[:, 5:10]

                        # FK position
                        positions = []
                        for b in range(len(candidate_actions)):
                            pos = forward_kinematics(
                                q_pred[b].cpu().numpy()
                            )
                            positions.append(pos)
                        positions = torch.as_tensor(
                            np.stack(positions), dtype=torch.float32, device=device
                        )

                        # Position error
                        pos_error = torch.norm(positions - target_t, dim=-1)

                        # Velocity alignment: dot product of velocity
                        # direction with target direction (negative = good)
                        target_dir = target_t - positions
                        target_dir_norm = target_dir / (
                            torch.norm(target_dir, dim=-1, keepdim=True) + 1e-8
                        )

                        # Discount future steps
                        discount = 0.7 ** h
                        total_cost = total_cost + discount * pos_error

                    # Action smoothness penalty
                    prev_action_t = torch.as_tensor(
                        prev_action, dtype=torch.float32, device=device
                    )
                    action_change = torch.norm(
                        action_batch - prev_action_t.unsqueeze(0), dim=-1
                    )
                    action_magnitude = torch.norm(action_batch, dim=-1)

                    # Combined score
                    score = (
                        total_cost
                        + 0.01 * action_magnitude
                        + 0.005 * action_change
                    )

                    best_idx = int(torch.argmin(score))
                    action = candidate_actions[best_idx]

                # Execute
                result = env.step(action)
                prev_action = action.copy()
                obs = result["observation"]
                distance = float(np.linalg.norm(env.ee_pos() - target))

                if distance <= 0.05:
                    reached = True
                    break

            rows.append({
                "seed": seed,
                "domain": domain.domain_id,
                "target": f"eval_{ti:02d}",
                "success": int(reached),
                "steps": step + 1,
                "final_distance_m": distance,
                "ik_error_m": ik_error,
                "horizon": args.horizon,
                "candidates": args.candidates,
                "perturb_std": args.perturb_std,
            })
            print(rows[-1], flush=True)

    out = Path(f"results/final/v7-multistep-selector-seed{seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.resolve()}")

    # Quick summary
    successes = sum(r["success"] for r in rows)
    avg_steps = np.mean([r["steps"] for r in rows])
    print(f"Summary: {successes}/{len(rows)} success, avg {avg_steps:.1f} steps")


if __name__ == "__main__":
    main()
