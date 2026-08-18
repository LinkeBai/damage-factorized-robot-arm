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
import csv
import json
from datetime import datetime, timezone
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
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.controllers import solve_reach_reference, joint_reference_action

PUSH_XML = "sim/assets/arm_push.xml"


def active_probe_action(step: int, sequence_index: int) -> np.ndarray:
    """Deterministic, complementary excitation for residual identification."""
    active_joint = (sequence_index + step // 20) % 5
    sign = 1.0 if ((step // 10) + sequence_index) % 2 == 0 else -1.0
    action = np.zeros(5, dtype=np.float64)
    action[active_joint] = 0.7 * sign
    action[(active_joint + 2) % 5] = 0.25 * np.sin(
        2.0 * np.pi * step / (25.0 + 5.0 * sequence_index)
    )
    return action


def collect_push_trajectory(
    domain: DomainSpec,
    *,
    steps: int,
    seed: int,
    target: np.ndarray,
    excitation: str = "random",
    sequence_index: int = 0,
    block_initial_xy: np.ndarray | None = None,
) -> SimTrajectory:
    env = MujocoArmEnv(
        xml_path=PUSH_XML,
        residual_physics=domain.residual,
        block_initial_xy=block_initial_xy,
    )
    obs = env.reset(target=target, damage_config=domain.damage)
    rng = np.random.default_rng(seed)
    action = np.zeros(5, dtype=np.float64)
    states = [obs["state"].copy()]
    commanded: list[np.ndarray] = []
    applied: list[np.ndarray] = []
    contact_steps = 0
    initial_block = env.block_pos().copy()
    locked = {i: domain.damage.lock_angle_of(i) for i in domain.damage.locked}
    approach_reference = push_reference = None
    if excitation == "goal":
        approach = np.array([initial_block[0] - 0.03, initial_block[1], 0.025])
        approach_reference, _ = solve_reach_reference(
            approach, env.joint_ranges, locked_joints=locked
        )
        push_reference, _ = solve_reach_reference(
            target, env.joint_ranges, locked_joints=locked
        )
    for step in range(steps):
        if excitation == "active":
            # Complementary square-wave probes expose motor scale, damping,
            # delay, deadband and reversal effects. K is nested over distinct
            # phase/joint combinations instead of repeated random walks.
            action = active_probe_action(step, sequence_index)
        elif excitation == "random":
            random_probe = rng.uniform(-0.45, 0.45, size=5)
            action = 0.75 * action + 0.25 * random_probe
        elif excitation == "goal":
            reference = approach_reference if step < steps * 0.4 else push_reference
            action = joint_reference_action(
                states[-1][:10],
                reference,
                locked_joints=tuple(domain.damage.locked),
            )
        else:
            raise ValueError(f"unknown excitation mode: {excitation}")
        action[domain.damage.locked] = 0.0
        result = env.step(action)
        commanded.append(action.copy())
        applied.append(env.last_applied_action)
        states.append(result["observation"]["state"].copy())
        contact_steps += int(env.has_contact("tool_geom", "block_geom"))
    return SimTrajectory(
        domain_id=domain.domain_id,
        states=torch.as_tensor(np.stack(states), dtype=torch.float32),
        actions=torch.as_tensor(np.stack(commanded), dtype=torch.float32),
        applied_actions=torch.as_tensor(np.stack(applied), dtype=torch.float32),
        metadata={
            "tool_block_contact_steps": contact_steps,
            "block_displacement_m": float(np.linalg.norm(env.block_pos() - initial_block)),
        },
    )


def collect_push_domains(
    domains: tuple[DomainSpec, ...],
    *,
    trajectories_per_domain: int,
    steps: int,
    seed: int,
    targets: tuple[np.ndarray, ...],
    excitation: str = "random",
    block_initial_xy: np.ndarray | None = None,
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
                    excitation=excitation,
                    sequence_index=ti,
                    block_initial_xy=block_initial_xy,
                )
            )
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. 7,17,27,42,51")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--calibration-excitation", choices=("random", "active"), default="active")
    ap.add_argument("--latent-steps", type=int, default=50)
    ap.add_argument("--gate-only", action="store_true", help="Evaluate only topology-only and DFWM")
    ap.add_argument("--latent-lr", type=float, default=0.01)
    ap.add_argument("--latent-l2", type=float, default=0.1)
    ap.add_argument("--latent-max-abs", type=float, default=1.0)
    ap.add_argument("--latent-patience", type=int, default=5)
    ap.add_argument("--train-active-probes", type=int, default=0)
    ap.add_argument("--split", type=Path, default=Path("config/splits/g1_5dof_v1.yaml"))
    ap.add_argument("--shots", type=str, default=None)
    ap.add_argument("--train-excitation", choices=("random", "goal"), default="random")
    ap.add_argument("--evaluation-excitation", choices=("random", "goal"), default="random")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--block-initial-xy", type=str, default="0.24,0.10")
    ap.add_argument("--target-split", type=Path, default=Path("config/splits/push_targets_5dof_v1.yaml"))
    ap.add_argument("--residual-supervision-weight", type=float, default=0.0)
    ap.add_argument("--residual-consistency-weight", type=float, default=0.0)
    ap.add_argument("--history-supervision-weight", type=float, default=0.0)
    ap.add_argument("--include-oracle", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else (args.seed,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or Path("runs") / "g1_push_6methods_5seeds" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "push_results.csv"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "device": str(device),
        "torch_version": torch.__version__,
        "push_xml": PUSH_XML,
        "methods": ["dfwm", "topology_only", "history_encoder", "parameter_matched", "monolithic_matched", "residual_only"],
        "calibration_shots": [0, 1, 2, 5],
        "calibration_excitation": args.calibration_excitation,
        "latent_steps": args.latent_steps,
        "gate_only": args.gate_only,
        "latent_lr": args.latent_lr,
        "latent_l2": args.latent_l2,
        "latent_max_abs": args.latent_max_abs,
        "latent_patience": args.latent_patience,
        "train_active_probes": args.train_active_probes,
        "split": str(args.split),
        "shots": args.shots,
        "train_excitation": args.train_excitation,
        "evaluation_excitation": args.evaluation_excitation,
        "steps": args.steps,
        "target_split": str(args.target_split),
        "residual_supervision_weight": args.residual_supervision_weight,
        "residual_consistency_weight": args.residual_consistency_weight,
        "history_supervision_weight": args.history_supervision_weight,
        "include_oracle": args.include_oracle,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Output: {output_dir}", flush=True)
    probe = MujocoArmEnv(xml_path=PUSH_XML)
    probe_obs = probe.reset(target=np.array([0.25, 0.15, 0.02]))
    print(f"Push state dim: {probe_obs['state'].shape[0]}", flush=True)

    all_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        protocol = load_g1_protocol(args.split)
        requested_shots = (
            tuple(int(value) for value in args.shots.split(","))
            if args.shots else protocol.calibration_shots
        )
        block_initial_xy = np.asarray(
            [float(value) for value in args.block_initial_xy.split(",")],
            dtype=np.float64,
        )
        split = load_target_split(args.target_split)
        cal = tuple(t.as_array() for t in split.calibration)
        targets = tuple(t.as_array() for t in split.evaluation)
        ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

        train = collect_push_domains(
            protocol.train, trajectories_per_domain=2, steps=args.steps,
            seed=seed * 10_000, targets=cal,
            excitation=args.train_excitation,
            block_initial_xy=block_initial_xy,
        )
        if args.train_active_probes:
            train += collect_push_domains(
                protocol.train,
                trajectories_per_domain=args.train_active_probes,
                steps=args.steps,
                seed=seed * 10_000 + 5_000,
                targets=cal,
                excitation="active",
                block_initial_xy=block_initial_xy,
            )
        models = train_mechanism_models(
            protocol.train,
            train,
            ranges,
            epochs=args.epochs,
            device=device,
            residual_supervision_weight=args.residual_supervision_weight,
            residual_consistency_weight=args.residual_consistency_weight,
            history_supervision_weight=args.history_supervision_weight,
        )
        print(f"seed {seed}: models trained", flush=True)

        for di, domain in enumerate(protocol.test):
            calib = collect_push_domains(
                (domain,), trajectories_per_domain=6, steps=args.steps,
                seed=seed * 100_000 + di * 1000, targets=cal,
                excitation=args.calibration_excitation,
                block_initial_xy=block_initial_xy,
            )
            evald = collect_push_domains(
                (domain,), trajectories_per_domain=3, steps=args.steps,
                seed=seed * 100_000 + di * 1000 + 500, targets=targets,
                excitation=args.evaluation_excitation,
                block_initial_xy=block_initial_xy,
            )
            rows = evaluate_test_domain(
                models, domain, calib, evald, ranges,
                shots=requested_shots,
                latent_steps=args.latent_steps,
                device=device,
                include_baselines=not args.gate_only,
                calibration_validation=[calib[-1]],
                latent_lr=args.latent_lr,
                latent_l2=args.latent_l2,
                latent_max_abs=args.latent_max_abs,
                latent_patience=args.latent_patience,
                include_oracle=args.include_oracle,
            )
            coverage = {
                "evaluation_contact_steps": sum(int(t.metadata.get("tool_block_contact_steps", 0)) for t in evald),
                "evaluation_block_displacement_m": float(np.mean([float(t.metadata.get("block_displacement_m", 0.0)) for t in evald])),
            }
            for row in rows:
                row.update(coverage)
            for r in rows:
                r["seed"] = seed
            all_rows += rows
        print(f"seed {seed}: done", flush=True)
        fieldnames = sorted({key for row in all_rows for key in row})
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"progress {len({int(r['seed']) for r in all_rows})}/{len(seeds)} seeds saved", flush=True)

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
