"""Run the complete simulation-only G0/G1 smoke pipeline.

This command intentionally stops short of claiming G0 completion: mesh physics
and hardware safety still require real-arm calibration. It does verify the
offline chain end to end:

1. load simple and full-mesh MuJoCo variants;
2. compute intact/D2/D3/D4 common reachable targets;
3. collect append-only trajectories under every morphology;
4. pretrain topology encoder + world model on nominal simulation;
5. freeze both and infer only z_residual on held-out actuator scaling;
6. write an auditable JSON summary and dataset manifest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from robotarm.data.schema import Episode, StepRecord
from robotarm.data.storage import EpisodeDataset
from robotarm.envs.damage import D2, D3, D4, DamageConfig
from robotarm.envs.fk import forward_kinematics
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.reachability import (
    analyze_damage_morphology,
    common_reachable_region,
    sample_targets_from_common,
)
from robotarm.models.residual_context import (
    LatentOptConfig,
    compose_context,
    latent_optimize,
)
from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel
from robotarm.training.g1_mechanism import rssm_training_loss

AXES = torch.tensor(
    [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)
DEPTH = torch.linspace(0.0, 1.0, 5)


def topology_embedding(
    encoder: TopologyEncoder,
    damage: DamageConfig,
    joint_ranges: np.ndarray,
) -> torch.Tensor:
    limits = torch.as_tensor(joint_ranges / np.pi, dtype=torch.float32)
    return encoder(
        torch.as_tensor(damage.joint_mask, dtype=torch.float32),
        torch.as_tensor(damage.lock_angle, dtype=torch.float32),
        AXES,
        limits,
        DEPTH,
    )


def rollout(
    env: MujocoArmEnv,
    target: np.ndarray,
    damage: DamageConfig,
    *,
    steps: int,
    seed: int,
    episode_id: str,
) -> tuple[Episode, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    obs = env.reset(target=target, damage_config=damage)
    records: list[StepRecord] = []
    states = [obs["state"].copy()]
    actions = []
    for step in range(steps):
        action = rng.uniform(-0.35, 0.35, size=5)
        action[damage.locked] = 0.0
        result = env.step(action)
        next_obs = result["observation"]
        done = step == steps - 1
        records.append(
            StepRecord(
                observation=obs,
                action_commanded=action.copy(),
                action_applied=action.copy(),
                next_observation=next_obs,
                reward=float(result["reward"]),
                success=bool(result["success"]),
                done=done,
                safety_flags={"valid": True},
                hardware_state={"model_variant": env.model_variant},
            )
        )
        actions.append(action)
        states.append(next_obs["state"].copy())
        obs = next_obs

    episode = Episode(
        episode_id=episode_id,
        timestamp_ns=time.time_ns(),
        platform="sim",
        task_id="reach",
        target_id=f"target_{episode_id}",
        split="calibration",
        damage_id=episode_id.split("_")[0],
        joint_mask=damage.joint_mask.copy(),
        lock_angle=damage.lock_angle.copy(),
        steps=records,
        seed=seed,
        config_hash="offline_pipeline_v1",
    )
    return (
        episode,
        torch.as_tensor(np.stack(states), dtype=torch.float32),
        torch.as_tensor(np.stack(actions), dtype=torch.float32),
    )


def sequence_nll(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    return wm.predict_multi_step(states, actions, context)["nll"].mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/offline_pipeline"))
    parser.add_argument("--reachability-samples", type=int, default=20_000)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--trajectory-steps", type=int, default=30)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--latent-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    dataset = EpisodeDataset(run_dir / "dataset", version="offline-smoke-v1")

    simple = MujocoArmEnv(model_variant="simple")
    mesh = MujocoArmEnv(model_variant="mesh")
    morphologies = [
        ("intact", DamageConfig.intact()),
        ("D2", D2()),
        ("D3", D3()),
        ("D4", D4()),
    ]

    reachability = [
        analyze_damage_morphology(
            name,
            forward_kinematics,
            simple.joint_ranges,
            damage,
            n=args.reachability_samples,
            rng=np.random.default_rng(args.seed + i),
        )
        for i, (name, damage) in enumerate(morphologies)
    ]
    grid, centers = common_reachable_region(reachability, voxel_size=0.03)
    targets = sample_targets_from_common(
        reachability,
        args.targets,
        voxel_size=0.03,
        rng=np.random.default_rng(args.seed + 20),
    )

    train_sequences: list[tuple[DamageConfig, torch.Tensor, torch.Tensor]] = []
    for i, ((name, damage), target) in enumerate(zip(morphologies, targets)):
        episode, states, actions = rollout(
            simple,
            target,
            damage,
            steps=args.trajectory_steps,
            seed=args.seed + 100 + i,
            episode_id=f"{name}_nominal_{i:02d}",
        )
        dataset.add(episode, source="sim")
        train_sequences.append((damage, states, actions))

    encoder = TopologyEncoder()
    wm = WorldModel()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(wm.parameters()), lr=3e-3
    )
    zero_residual = torch.zeros(8)
    initial_loss = None
    final_loss = None
    for train_step in range(args.train_steps):
        optimizer.zero_grad()
        losses = []
        for damage, states, actions in train_sequences:
            topology = topology_embedding(encoder, damage, simple.joint_ranges)
            context = compose_context(
                topology, zero_residual, context_dim=wm.cfg.context_dim
            )
            losses.append(
                rssm_training_loss(
                    wm,
                    states.unsqueeze(0),
                    actions.unsqueeze(0),
                    context.unsqueeze(0),
                )
            )
        loss = torch.stack(losses).mean()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(wm.parameters()), max_norm=10.0
        )
        optimizer.step()
        final_loss = float(loss.detach())

    # Held-out residual physics: same D3 topology, 65% actuator effectiveness.
    residual_scale = np.array([1.5, 1.8, 2.4, 1.8, 3.0]) * 0.65
    residual_env = MujocoArmEnv(ctrl_scale=residual_scale)
    calibration_states = []
    calibration_actions = []
    for k in range(3):
        _, states, actions = rollout(
            residual_env,
            targets[(k + 4) % len(targets)],
            D3(),
            steps=args.trajectory_steps,
            seed=args.seed + 200 + k,
            episode_id=f"D3_residual_{k:02d}",
        )
        calibration_states.append(states)
        calibration_actions.append(actions)
    states_k = torch.stack(calibration_states)
    actions_k = torch.stack(calibration_actions)
    topology = topology_embedding(encoder, D3(), simple.joint_ranges).detach()
    context_zero = compose_context(
        topology, torch.zeros(8), context_dim=wm.cfg.context_dim
    )
    with torch.no_grad():
        nll_before = float(
            torch.stack(
                [
                    sequence_nll(wm, states_k[k], actions_k[k], context_zero)
                    for k in range(states_k.shape[0])
                ]
            ).mean()
        )
    residual = latent_optimize(
        wm,
        topology,
        states_k,
        actions_k,
        LatentOptConfig(d=8, steps=args.latent_steps, lr=0.1),
    )
    context_adapted = compose_context(
        topology, residual.z.detach(), context_dim=wm.cfg.context_dim
    )
    with torch.no_grad():
        nll_after = float(
            torch.stack(
                [
                    sequence_nll(wm, states_k[k], actions_k[k], context_adapted)
                    for k in range(states_k.shape[0])
                ]
            ).mean()
        )

    summary = {
        "status": "offline_smoke_complete",
        "g0_passed": False,
        "g0_blocker": "real-arm kinematics and safety calibration pending",
        "models": {
            "simple": {"nq": simple.model.nq, "nu": simple.model.nu},
            "mesh": {
                "nq": mesh.model.nq,
                "nu": mesh.model.nu,
                "nmesh": mesh.model.nmesh,
            },
        },
        "damage_mapping": {"D2": "j2", "D3": "j3", "D4": "j4"},
        "common_reachable_voxels": int(grid.sum()),
        "common_target_count": int(len(targets)),
        "dataset_integrity": dataset.verify_integrity(),
        "world_model": {
            "train_nll_initial": initial_loss,
            "train_nll_final": final_loss,
            "residual_nll_before": nll_before,
            "residual_nll_after": nll_after,
            "residual_norm": float(residual.z.detach().norm()),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
