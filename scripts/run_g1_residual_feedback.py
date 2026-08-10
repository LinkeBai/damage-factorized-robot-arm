"""Evaluate a conservative residual feedback layer over the verified IK+PD baseline."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import (
    jacobian_reach_action,
    joint_reference_action,
    solve_reach_reference,
)
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    protocol = build_g1_protocol()
    targets = tuple(t.as_array() for t in load_target_split().evaluation)
    rows = []
    for domain in protocol.test:
        env = MujocoArmEnv(residual_physics=domain.residual)
        locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
        for ti, target in enumerate(targets):
            reference, ik_error = solve_reach_reference(target, env.joint_ranges, locked_joints=locked)
            obs = env.reset(target=target, damage_config=domain.damage)
            reached = False
            for step in range(300):
                pd = joint_reference_action(obs["state"], reference, locked_joints=tuple(domain.damage.locked))
                residual = jacobian_reach_action(obs["state"], target, locked_joints=tuple(domain.damage.locked))
                action = np.clip(pd + 0.25 * residual, -1.0, 1.0)
                result = env.step(action)
                obs = result["observation"]
                distance = float(np.linalg.norm(env.ee_pos() - target))
                if distance <= 0.05:
                    reached = True
                    break
            rows.append({
                "domain": domain.domain_id, "target": f"eval_{ti:02d}",
                "ik_error_m": ik_error, "success": int(reached),
                "steps": step + 1, "final_distance_m": distance,
                "residual_gain": 0.25,
            })
            print(rows[-1], flush=True)
    out = Path("results/final/g1-residual-feedback.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
