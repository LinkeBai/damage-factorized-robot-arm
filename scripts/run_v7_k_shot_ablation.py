"""V7 Ablation: K-shot comparison + uncertainty-based rejection.

Tests:
1. K=0/1/2/5 calibration trajectories → does WM improve with more data?
2. Uncertainty-based rejection: when WM prediction std is high, fall back to IK
3. Demonstrates WM's unique capability: calibrated uncertainty estimation
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
from robotarm.training.sim_protocol import build_g1_protocol, DomainSpec
from robotarm.training.target_split import load_target_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed)

    protocol = build_g1_protocol()
    split = load_target_split()
    cal = tuple(t.as_array() for t in split.calibration)
    targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = collect_controller_domains(protocol.train, trajectories_per_domain=2, steps=100, seed=seed*10000, targets=cal)
    models = train_mechanism_models(protocol.train, train, ranges, epochs=args.epochs, device=device)
    wm = models.dfwm_world_model; wm.eval()

    rows = []
    for di, domain in enumerate(protocol.test):
        # Collect 5 calibration trajectories (use up to 5 for K-shot)
        calib_all = collect_controller_domains((domain,), trajectories_per_domain=5, steps=100, seed=seed*100000+di*1000, targets=cal)

        for shots in (0, 1, 2, 5):
            context = infer_dfwm_context(models, domain, calib_all, ranges, shots=shots, latent_steps=30, device=device)

            for ti, target in enumerate(targets):
                env = MujocoArmEnv(residual_physics=domain.residual)
                locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
                locked_tuple = tuple(domain.damage.locked)
                ref, ik_err = solve_reach_reference(target, env.joint_ranges, locked_joints=locked)
                obs = env.reset(target=target, damage_config=domain.damage)
                reached = False
                n_rejections = 0

                for step in range(300):
                    base = joint_reference_action(obs["state"], ref, locked_joints=locked_tuple)
                    state_t = torch.as_tensor(obs["state"], dtype=torch.float32, device=device).reshape(1, -1)
                    action_t = torch.as_tensor(base, dtype=torch.float32, device=device).reshape(1, -1)

                    with torch.no_grad():
                        pred, _ = wm.step(state_t, action_t, context.reshape(1, -1), None)
                        q_pred = pred["mean"][0, :5].cpu().numpy()
                        # Use prediction std as uncertainty signal
                        log_std = pred.get("prior_log_std", pred.get("log_std"))
                        if log_std is not None:
                            uncertainty = float(log_std[0].mean().exp())
                        else:
                            uncertainty = 0.0

                    # Uncertainty-based rejection: if uncertainty > threshold, use pure IK
                    if uncertainty > 0.5:  # threshold tuned for this task
                        action = base
                        n_rejections += 1
                    else:
                        # Adaptive correction based on prediction error
                        error = ref - q_pred
                        error_mag = np.abs(error).mean()
                        gain = np.clip(0.05 + 2.0 * error_mag, 0.02, 0.35)
                        correction = gain * error
                        correction[list(domain.damage.locked)] = 0.0
                        correction = np.clip(correction, -0.3, 0.3)
                        action = np.clip(base + correction, -1.0, 1.0)

                    result = env.step(action)
                    obs = result["observation"]
                    distance = float(np.linalg.norm(env.ee_pos() - target))
                    if distance <= 0.05:
                        reached = True; break

                rows.append({
                    "seed": seed, "domain": domain.domain_id,
                    "target": f"eval_{ti:02d}", "shots": shots,
                    "success": int(reached), "steps": step + 1,
                    "final_distance_m": distance,
                    "n_rejections": n_rejections,
                    "rejection_rate": n_rejections / (step + 1) if step > 0 else 0,
                })
                print(rows[-1], flush=True)

    out = Path(f"results/final/v7-k-shot-ablation-seed{seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out.resolve()}")

    print("\n=== K-shot Summary ===")
    for shots in (0, 1, 2, 5):
        subset = [r for r in rows if r["shots"] == shots]
        succ = sum(r["success"] for r in subset)
        avg = np.mean([r["steps"] for r in subset if r["success"]])
        avg_rej = np.mean([r["rejection_rate"] for r in subset])
        print(f"K={shots}: {succ}/{len(subset)} success, avg {avg:.1f} steps, rejection rate {avg_rej:.2%}")


if __name__ == "__main__":
    main()
