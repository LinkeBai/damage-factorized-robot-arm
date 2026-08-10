"""Evaluate a gated one-step world-model correction over IK+PD."""
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
    args = ap.parse_args()
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    protocol = build_g1_protocol(); split = load_target_split()
    calibration_targets = tuple(t.as_array() for t in split.calibration)
    eval_targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges
    train_data = collect_controller_domains(
        protocol.train, trajectories_per_domain=2, steps=100,
        seed=seed * 10_000, targets=calibration_targets
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = train_mechanism_models(protocol.train, train_data, ranges, epochs=60, device=device)
    rows = []
    for di, domain in enumerate(protocol.test):
        calibration = collect_controller_domains(
            (domain,), trajectories_per_domain=5, steps=100,
            seed=seed * 100_000 + di * 1_000, targets=calibration_targets
        )
        for shots in (0, 5):
            context = infer_dfwm_context(models, domain, calibration, ranges,
                                         shots=shots, latent_steps=30, device=device)
            wm = models.dfwm_world_model
            for ti, target in enumerate(eval_targets):
                env = MujocoArmEnv(residual_physics=domain.residual)
                locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
                reference, ik_error = solve_reach_reference(target, env.joint_ranges, locked_joints=locked)
                obs = env.reset(target=target, damage_config=domain.damage)
                reached = False
                for step in range(300):
                    base = joint_reference_action(obs["state"], reference, locked_joints=tuple(domain.damage.locked))
                    state = torch.as_tensor(obs["state"], dtype=torch.float32, device=device).reshape(1, -1)
                    action = torch.as_tensor(base, dtype=torch.float32, device=device).reshape(1, -1)
                    with torch.no_grad():
                        prediction, _ = wm.step(state, action, context.reshape(1, -1), None)
                    predicted_q = prediction["mean"][0, :5].cpu().numpy()
                    correction = np.clip(0.15 * (reference - predicted_q), -0.15, 0.15)
                    correction[list(domain.damage.locked)] = 0.0
                    result = env.step(np.clip(base + correction, -1.0, 1.0))
                    obs = result["observation"]
                    distance = float(np.linalg.norm(env.ee_pos() - target))
                    if distance <= 0.05:
                        reached = True; break
                rows.append({"seed": seed, "domain": domain.domain_id, "target": f"eval_{ti:02d}",
                             "shots": shots, "success": int(reached), "steps": step + 1,
                             "final_distance_m": distance, "ik_error_m": ik_error})
                print(rows[-1], flush=True)
    out = Path(f"results/final/g1-worldmodel-hybrid-seed{seed}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
