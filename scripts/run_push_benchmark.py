"""Push-task prediction benchmark: does DFWM recover its advantage?

Push (contact dynamics) makes residual physics (friction, actuator loss,
backlash) affect the state far more than free-space Reach. This script reuses
the DFWM training/eval machinery but collects trajectories with the block-aware
arm_push.xml environment, so the state is 14-D (5 qpos + 5 qvel + block pos/vel).

The hypothesis: with a harder residual channel, factorizing topology vs residual
shows a real prediction advantage over topology-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.g1_mechanism import (
    evaluate_test_domain,
    train_mechanism_models,
)
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import build_g1_protocol, DomainSpec
from robotarm.training.target_split import load_target_split

PUSH_XML = "sim/assets/arm_push.xml"


def collect_push_trajectory(
    domain: DomainSpec, *, steps: int, seed: int, target: np.ndarray
) -> SimTrajectory:
    env = MujocoArmEnv(xml_path=PUSH_XML, residual_physics=domain.residual)
    obs = env.reset(target=target, damage_config=domain.damage)
    rng = np.random.default_rng(seed)
    action = np.zeros(5, dtype=np.float64)
    states = [obs["state"].copy()]
    commanded: list[np.ndarray] = []
    applied: list[np.ndarray] = []
    for _ in range(steps):
        excitation = rng.uniform(-0.45, 0.45, size=5)
        action = 0.75 * action + 0.25 * excitation
        action[domain.damage.locked] = 0.0
        result = env.step(action)
        commanded.append(action.copy())
        applied.append(env.last_applied_action)
        states.append(result["observation"]["state"].copy())
    return SimTrajectory(
        domain_id=domain.domain_id,
        states=torch.as_tensor(np.stack(states), dtype=torch.float32),
        actions=torch.as_tensor(np.stack(commanded), dtype=torch.float32),
        applied_actions=torch.as_tensor(np.stack(applied), dtype=torch.float32),
    )


def collect_push_domains(
    domains: tuple[DomainSpec, ...],
    *,
    trajectories_per_domain: int,
    steps: int,
    seed: int,
    targets: tuple[np.ndarray, ...],
) -> list[SimTrajectory]:
    trajs = []
    for di, domain in enumerate(domains):
        for ti in range(trajectories_per_domain):
            trajs.append(
                collect_push_trajectory(
                    domain,
                    steps=steps,
                    seed=seed + di * 1000 + ti,
                    target=targets[(di + ti) % len(targets)],
                )
            )
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. 7,17,27,42,51")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else (args.seed,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = MujocoArmEnv(xml_path=PUSH_XML)
    probe_obs = probe.reset(target=np.array([0.25, 0.15, 0.02]))
    print(f"Push state dim: {probe_obs['state'].shape[0]}", flush=True)

    all_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        protocol = build_g1_protocol()
        split = load_target_split()
        cal = tuple(t.as_array() for t in split.calibration)
        targets = tuple(t.as_array() for t in split.evaluation)
        ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

        train = collect_push_domains(
            protocol.train, trajectories_per_domain=2, steps=100,
            seed=seed * 10_000, targets=cal,
        )
        models = train_mechanism_models(
            protocol.train, train, ranges, epochs=args.epochs, device=device
        )
        print(f"seed {seed}: models trained", flush=True)

        for di, domain in enumerate(protocol.test):
            calib = collect_push_domains(
                (domain,), trajectories_per_domain=5, steps=100,
                seed=seed * 100_000 + di * 1000, targets=cal,
            )
            evald = collect_push_domains(
                (domain,), trajectories_per_domain=3, steps=100,
                seed=seed * 100_000 + di * 1000 + 500, targets=targets,
            )
            rows = evaluate_test_domain(
                models, domain, calib, evald, ranges,
                shots=protocol.calibration_shots, latent_steps=50, device=device,
            )
            for r in rows:
                r["seed"] = seed
            all_rows += rows
        print(f"seed {seed}: done", flush=True)

    # Summarize K=5 one-step and multi-step RMSE
    print("\n=== K=5 results (Push, {} seeds) ===".format(len(seeds)))
    for model in ("dfwm", "topology_only", "history_encoder", "parameter_matched",
                  "monolithic_matched", "residual_only"):
        one = [float(r["eval_rmse"]) for r in all_rows if r["model"] == model and int(r["shots"]) == 5]
        multi = [float(r["multi_step_rmse"]) for r in all_rows
                 if r["model"] == model and int(r["shots"]) == 5 and "multi_step_rmse" in r]
        if one:
            m = f"  multi-step={np.mean(multi):.4f}" if multi else ""
            print(f"{model:20s} one-step={np.mean(one):.4f}{m}")


if __name__ == "__main__":
    main()
