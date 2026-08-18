"""Measure whether fixed probes separate residual physics in the Push scene."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.sim_protocol import damage_from_name
from robotarm.envs.residual_physics import residual_profile
from run_push_benchmark import PUSH_XML, active_probe_action


def rollout(topology: str, residual: str, sequence: int, steps: int) -> tuple[np.ndarray, float, int]:
    env = MujocoArmEnv(xml_path=PUSH_XML, residual_physics=residual_profile(residual))
    obs = env.reset(target=np.array([0.25, 0.15, 0.02]), damage_config=damage_from_name(topology))
    start = env.block_pos().copy()
    states = [obs["state"].copy()]
    contacts = 0
    for step in range(steps):
        action = active_probe_action(step, sequence)
        action[env.damage_config.locked] = 0.0
        obs = env.step(action)["observation"]
        states.append(obs["state"].copy())
        contacts += int(env.has_contact("tool_geom", "block_geom"))
    return np.stack(states), float(np.linalg.norm(env.block_pos() - start)), contacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    rows = []
    for topology in ("D2", "D3"):
        for sequence in range(5):
            nominal, nominal_displacement, nominal_contacts = rollout(topology, "nominal", sequence, args.steps)
            for residual in ("weak_motor", "high_damping", "delay_1", "mixed_composition", "mixed_unseen"):
                states, displacement, contacts = rollout(topology, residual, sequence, args.steps)
                rows.append({
                    "topology": topology,
                    "sequence": sequence,
                    "residual": residual,
                    "trajectory_rmse_vs_nominal": float(np.sqrt(np.mean((states - nominal) ** 2))),
                    "block_displacement_m": displacement,
                    "tool_block_contact_steps": contacts,
                    "nominal_block_displacement_m": nominal_displacement,
                    "nominal_tool_block_contact_steps": nominal_contacts,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
