"""V7 full evaluation: adaptive hybrid on standard + extended (hard) domains.

Demonstrates WM value across increasing difficulty: D2/D3 (standard) and
D4 + mixed_unseen (hard). Hypothesis: WM gain scales with problem difficulty
because IK+PD's fixed gains become suboptimal under severe residual physics.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.control_eval import infer_dfwm_context
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import DomainSpec, build_g1_protocol
from robotarm.training.target_split import load_target_split

EXTENDED_DOMAINS = [
    DomainSpec("D2", "mixed_unseen", "test"),
    DomainSpec("D3", "mixed_unseen", "test"),
    DomainSpec("D4", "mixed_composition", "test"),
    DomainSpec("D4", "mixed_unseen", "test"),
]


def evaluate_ik_pd(domain, targets):
    """Baseline: pure IK+PD."""
    results = []
    env = MujocoArmEnv(residual_physics=domain.residual)
    locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
    locked_tuple = tuple(domain.damage.locked)
    for ti, target in enumerate(targets):
        ref, ik_err = solve_reach_reference(target, env.joint_ranges, locked_joints=locked)
        obs = env.reset(target=target, damage_config=domain.damage)
        reached = False
        for step in range(300):
            action = joint_reference_action(obs["state"], ref, locked_joints=locked_tuple)
            result = env.step(action)
            obs = result["observation"]
            distance = float(np.linalg.norm(env.ee_pos() - target))
            if distance <= 0.05:
                reached = True; break
        results.append({"success": int(reached), "steps": step + 1, "final_distance_m": distance})
    return results


def evaluate_wm_hybrid(domain, targets, wm, context, device):
    """V7 adaptive WM-hybrid with multi-step correction."""
    results = []
    env = MujocoArmEnv(residual_physics=domain.residual)
    locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
    locked_tuple = tuple(domain.damage.locked)
    for ti, target in enumerate(targets):
        ref, ik_err = solve_reach_reference(target, env.joint_ranges, locked_joints=locked)
        obs = env.reset(target=target, damage_config=domain.damage)
        reached = False
        prev_correction = np.zeros(5)
        for step in range(300):
            base = joint_reference_action(obs["state"], ref, locked_joints=locked_tuple)
            state_t = torch.as_tensor(obs["state"], dtype=torch.float32, device=device).reshape(1, -1)
            action_t = torch.as_tensor(base, dtype=torch.float32, device=device).reshape(1, -1)
            with torch.no_grad():
                pred1, hidden = wm.step(state_t, action_t, context.reshape(1, -1), None)
                q1 = pred1["mean"][0, :5].cpu().numpy()
                pred2, _ = wm.step(pred1["mean"], action_t, context.reshape(1, -1), hidden)
                q2 = pred2["mean"][0, :5].cpu().numpy()
            qvel = obs["state"][5:]
            error_1step = ref - q1
            error_2step = ref - q2
            position_error = 0.7 * error_1step + 0.3 * error_2step
            error_mag = np.abs(position_error)
            adaptive_gain = np.clip(0.08 + 1.5 * error_mag, 0.05, 0.35)
            ref_direction = ref - obs["state"][:5]
            ref_norm = np.linalg.norm(ref_direction) + 1e-8
            vel_alignment = np.sum(qvel * ref_direction) / ref_norm
            vel_factor = np.clip(1.0 - 0.5 * vel_alignment, 0.5, 1.5)
            correction = adaptive_gain * vel_factor * position_error
            correction = 0.7 * correction + 0.3 * prev_correction
            correction = np.clip(correction, -0.3, 0.3)
            correction[list(domain.damage.locked)] = 0.0
            prev_correction = correction.copy()
            action = np.clip(base + correction, -1.0, 1.0)
            result = env.step(action)
            obs = result["observation"]
            distance = float(np.linalg.norm(env.ee_pos() - target))
            if distance <= 0.05:
                reached = True; break
        results.append({"success": int(reached), "steps": step + 1, "final_distance_m": distance})
    return results


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

    t0 = time.perf_counter()
    train = collect_controller_domains(protocol.train, trajectories_per_domain=2, steps=100, seed=seed*10000, targets=cal)
    models = train_mechanism_models(protocol.train, train, ranges, epochs=args.epochs, device=device)
    print(f"Training: {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    all_domains = list(protocol.test) + EXTENDED_DOMAINS

    for di, domain in enumerate(all_domains):
        calib = collect_controller_domains((domain,), trajectories_per_domain=5, steps=100, seed=seed*100000+di*1000, targets=cal)
        context = infer_dfwm_context(models, domain, calib, ranges, shots=5, latent_steps=30, device=device)
        wm = models.dfwm_world_model; wm.eval()

        # IK baseline
        ik = evaluate_ik_pd(domain, targets)
        # WM hybrid
        wm_res = evaluate_wm_hybrid(domain, targets, wm, context, device)

        for ti in range(len(targets)):
            for method, res in [("ik_pd", ik[ti]), ("wm_hybrid", wm_res[ti])]:
                rows.append({
                    "seed": seed, "domain": domain.domain_id,
                    "target": f"eval_{ti:02d}", "method": method,
                    "success": res["success"], "steps": res["steps"],
                    "final_distance_m": res["final_distance_m"],
                })
        ik_avg = np.mean([r["steps"] for r in ik if r["success"]])
        wm_avg = np.mean([r["steps"] for r in wm_res if r["success"]])
        ik_succ = sum(r["success"] for r in ik)
        wm_succ = sum(r["success"] for r in wm_res)
        print(f"{domain.domain_id}: IK={ik_succ}/4 avg={ik_avg:.1f} | WM={wm_succ}/4 avg={wm_avg:.1f} | delta={wm_avg-ik_avg:+.1f}", flush=True)

    out = Path(f"results/final/v7-full-eval-seed{seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out.resolve()}")

    # Summary
    for domain_id in sorted(set(r["domain"] for r in rows)):
        for method in ("ik_pd", "wm_hybrid"):
            subset = [r for r in rows if r["domain"] == domain_id and r["method"] == method]
            succ = sum(r["success"] for r in subset)
            avg = np.mean([r["steps"] for r in subset if r["success"]])
            print(f"  {domain_id:30s} {method:10s}: {succ}/{len(subset)} avg={avg:.1f}")


if __name__ == "__main__":
    main()
