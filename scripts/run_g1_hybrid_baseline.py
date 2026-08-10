"""Evaluate the first G1 hybrid baseline: topology-aware IK + joint PD."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    protocol = build_g1_protocol()
    split = load_target_split()
    targets = tuple(target.as_array() for target in split.evaluation)
    rows = []
    for domain in protocol.test:
        env = MujocoArmEnv(residual_physics=domain.residual)
        locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
        for target_index, target in enumerate(targets):
            reference, ik_error = solve_reach_reference(
                target, env.joint_ranges, locked_joints=locked
            )
            observation = env.reset(target=target, damage_config=domain.damage)
            reached = False
            final_distance = float("inf")
            for step in range(300):
                result = env.step(
                    joint_reference_action(
                        observation["state"], reference,
                        locked_joints=tuple(domain.damage.locked),
                    )
                )
                observation = result["observation"]
                final_distance = float(np.linalg.norm(env.ee_pos() - target))
                if final_distance <= 0.05:
                    reached = True
                    break
            rows.append({
                "domain": domain.domain_id,
                "target": f"eval_{target_index:02d}",
                "ik_error_m": ik_error,
                "success": int(reached),
                "steps": step + 1,
                "final_distance_m": final_distance,
            })
            print(rows[-1], flush=True)
    out = Path("results/final/g1-hybrid-baseline.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
