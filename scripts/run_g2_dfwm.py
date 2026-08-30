"""DFWM training and evaluation with proper z_residual learning.

KEY FIX vs prior attempts:
  Training: each domain has MULTIPLE physics profiles. For each training
  trajectory, z_residual is optimized on the calibration split of that
  trajectory before the world model gradient step. This forces the WM to
  learn "same topology, different z -> different dynamics".

  Evaluation: K goal-directed calibration trajectories -> latent_optimize
  z_residual -> evaluate on held-out evaluation trajectories.

Usage:
  python scripts/run_g2_dfwm.py --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.robotarm.envs.mujoco_env import MujocoArmEnv
from src.robotarm.models.topology_encoder import TopologyEncoder, TopologyEncoderConfig
from src.robotarm.models.world_model import WorldModel, WorldModelConfig
from src.robotarm.models.residual_context import (
    LatentOptConfig,
    ResidualContext,
    compose_context,
    latent_optimize,
)
from src.robotarm.training.g1_mechanism import (
    TOPOLOGY_DIM,
    RESIDUAL_DIM,
    encode_damage_batch,
    rssm_training_loss,
)
from src.robotarm.training.sim_protocol import load_g1_protocol
from src.robotarm.training.target_split import load_target_split
from src.robotarm.training.sim_data import SimTrajectory
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_dfwm_v1.yaml")
CONTEXT_DIM = TOPOLOGY_DIM + RESIDUAL_DIM  # 64 + 8 = 72


# ── helpers ───────────────────────────────────────────────────────────────────

def build_wm(device: torch.device) -> tuple[TopologyEncoder, WorldModel]:
    encoder = TopologyEncoder(TopologyEncoderConfig(out_dim=TOPOLOGY_DIM)).to(device)
    wm = WorldModel(WorldModelConfig(state_dim=14, action_dim=5, context_dim=CONTEXT_DIM)).to(device)
    return encoder, wm


def get_topology_context(
    encoder: TopologyEncoder,
    damages: list,
    joint_ranges: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return encode_damage_batch(encoder, damages, joint_ranges, device)


def multistep_rmse(
    wm: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    horizon: int = 10,
) -> float:
    horizon = min(horizon, actions.shape[1])
    sq_errs = []
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred = states[:, start].clone()
        hidden = None
        for h in range(horizon):
            out, hidden = wm.step(pred, actions[:, start + h], context, hidden)
            pred = out["mean"]
            sq_errs.append((pred - states[:, start + h + 1]).pow(2).mean(dim=-1))
    return float(torch.stack(sq_errs).mean().sqrt())


# ── training ──────────────────────────────────────────────────────────────────

def train_dfwm(
    trajectories_by_domain: dict[str, list[SimTrajectory]],
    joint_ranges: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    latent_cfg: LatentOptConfig,
) -> tuple[TopologyEncoder, WorldModel]:
    """Train WM where z_residual is optimized per-trajectory during training.

    For each epoch, for each domain's trajectories:
    1. Compute fixed e_topology from encoder.
    2. For each trajectory, optimize z_residual on first half (calibration).
    3. Use [e_topology, z_residual] as context for WM gradient update on
       second half (evaluation).

    This teaches the WM: "given this topology AND this z, predict these dynamics."
    """
    torch.manual_seed(seed)
    encoder, wm = build_wm(device)
    params = list(encoder.parameters()) + list(wm.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)

    domains = list(trajectories_by_domain.keys())
    all_trajs = list(trajectories_by_domain.items())

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        np.random.shuffle(all_trajs)

        for domain_id, trajs in all_trajs:
            if not trajs:
                continue

            # Get topology descriptor (gradient flows through encoder)
            damage = trajs[0].damage if hasattr(trajs[0], "damage") else None
            # domain_id format: "D2__nominal" -> topology = "D2"
            topology_name = domain_id.split("__")[0]
            from src.robotarm.training.sim_protocol import damage_from_name
            dmg = damage_from_name(topology_name)
            e_topo = encode_damage_batch(encoder, [dmg], joint_ranges, device)  # (1, 64)

            for traj in trajs:
                T = traj.states.shape[0] - 1
                if T < 4:
                    continue
                split = T // 2
                states = traj.states.to(device)
                actions = traj.actions.to(device)

                # --- z_residual optimization on first half ---
                # Use torch.no_grad context around latent_optimize to avoid
                # accumulating WM computation graphs in memory.
                with torch.no_grad():
                    e_topo_fixed = encode_damage_batch(encoder, [dmg], joint_ranges, device).squeeze(0).detach()

                rc = latent_optimize(
                    wm,
                    e_topo_fixed,
                    states[:split + 1].unsqueeze(0),
                    actions[:split].unsqueeze(0),
                    latent_cfg,
                )
                z = rc.z.detach().clone()  # detach completely

                # --- WM gradient update on second half ---
                # Recompute e_topo fresh each trajectory so graph is independent
                e_topo_fresh = encode_damage_batch(encoder, [dmg], joint_ranges, device)
                context = compose_context(
                    e_topo_fresh.squeeze(0),
                    z,
                    context_dim=CONTEXT_DIM,
                ).unsqueeze(0)  # (1, 72)

                states_eval = states[split:].unsqueeze(0)   # (1, T2+1, 14)
                actions_eval = actions[split:].unsqueeze(0) # (1, T2, 5)

                loss = rssm_training_loss(wm, states_eval, actions_eval, context)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_loss += float(loss)
                n_batches += 1

                # Explicitly free tensors to avoid memory accumulation
                del states, actions, states_eval, actions_eval, context, loss, rc
                torch.cuda.empty_cache() if device.type == "cuda" else None

        if (epoch + 1) % 5 == 0:
            avg = epoch_loss / max(n_batches, 1)
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}", flush=True)

    return encoder, wm


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_dfwm(
    encoder: TopologyEncoder,
    wm: WorldModel,
    domain,
    calibration_trajs: list[SimTrajectory],
    evaluation_trajs: list[SimTrajectory],
    joint_ranges: np.ndarray,
    device: torch.device,
    *,
    k: int,
    latent_cfg: LatentOptConfig,
    horizon: int = 10,
) -> dict[str, float]:
    """Evaluate with K calibration trajectories -> z_residual -> predict."""
    encoder.eval()
    wm.eval()

    topology_name = domain.domain_id.split("__")[0]
    from src.robotarm.training.sim_protocol import damage_from_name
    dmg = damage_from_name(topology_name)

    with torch.no_grad():
        e_topo = encode_damage_batch(encoder, [dmg], joint_ranges, device).squeeze(0)  # (64,)

    # K=0: use z=0
    if k == 0 or len(calibration_trajs) == 0:
        z = torch.zeros(RESIDUAL_DIM, device=device)
    else:
        cal = calibration_trajs[:k]
        cal_states = torch.stack([t.states for t in cal]).to(device)   # (K, T+1, 14)
        cal_actions = torch.stack([t.actions for t in cal]).to(device) # (K, T, 5)
        rc = latent_optimize(wm, e_topo, cal_states, cal_actions, latent_cfg)
        z = rc.z.detach()

    context = compose_context(e_topo, z, context_dim=CONTEXT_DIM).unsqueeze(0)  # (1, 72)

    eval_states = torch.stack([t.states for t in evaluation_trajs]).to(device)
    eval_actions = torch.stack([t.actions for t in evaluation_trajs]).to(device)

    # expand context to batch
    B = eval_states.shape[0]
    ctx = context.expand(B, -1)

    wm.eval()
    with torch.no_grad():
        rmse = multistep_rmse(wm, eval_states, eval_actions, ctx, horizon)

    return {
        "k": k,
        "ensemble_rmse": rmse,
        "z_norm": float(z.norm().item()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in frozen seeds {config['seeds']}")

    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    cal_shots = list(config["calibration_shots"])
    horizon = int(config["rollout_horizon"])

    output_dir = args.output_dir or (
        Path("runs/g2_dfwm") / f"seed{args.seed}_v1"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  epochs={epochs}  device={device}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration_targets = tuple(item.as_array() for item in targets.calibration)
    evaluation_targets = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    latent_cfg = LatentOptConfig(
        d=int(config["residual_dim"]),
        lr=float(config["latent_opt_lr"]),
        steps=int(config["latent_opt_steps"]),
        l2=float(config["latent_opt_l2"]),
    )

    # ── collect training trajectories via GPU batch (warp) ───────────────────
    print("\n[train] collecting trajectories (mujoco-warp GPU batch) …", flush=True)
    t_collect = time.perf_counter()
    all_train_trajs = collect_push_domains_warp(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        block_initial_xy=block_initial_xy,
    )
    # Group by domain_id for training
    trajs_by_domain: dict[str, list[SimTrajectory]] = {}
    tpd = int(config["trajectories_per_train_domain"])
    for i, domain in enumerate(protocol.train):
        trajs_by_domain[domain.domain_id] = all_train_trajs[i * tpd:(i + 1) * tpd]
        print(f"  {domain.domain_id}: {len(trajs_by_domain[domain.domain_id])} trajs", flush=True)
    print(f"  collect done in {time.perf_counter() - t_collect:.1f}s", flush=True)

    # ── train ─────────────────────────────────────────────────────────────────
    print("\n[train] DFWM with z_residual learning …", flush=True)
    t0 = time.perf_counter()
    encoder, wm = train_dfwm(
        trajs_by_domain, ranges,
        epochs=epochs, device=device, seed=args.seed,
        latent_cfg=latent_cfg,
    )
    train_secs = time.perf_counter() - t0
    print(f"  done in {train_secs:.1f}s", flush=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    rows = []
    for idx, domain in enumerate(protocol.test):
        print(f"\n[eval] {domain.domain_id} …", flush=True)

        # collect calibration trajectories (goal-directed)
        cal_trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=max(cal_shots),
            steps=steps,
            seed=args.seed * 100_000 + idx * 1000,
            targets=calibration_targets,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )

        # collect evaluation trajectories (separate seed)
        eval_trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + idx * 1000 + 500,
            targets=evaluation_targets,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )

        for k in cal_shots:
            metrics = evaluate_dfwm(
                encoder, wm, domain,
                cal_trajs, eval_trajs, ranges, device,
                k=k, latent_cfg=latent_cfg, horizon=horizon,
            )
            row = {
                "domain": domain.domain_id,
                "method": "dfwm",
                "seed": args.seed,
                **metrics,
            }
            rows.append(row)
            print(f"  K={k}: rmse={metrics['ensemble_rmse']:.4f}  z_norm={metrics['z_norm']:.3f}", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config_version": config["version"],
        "method": "dfwm",
        "seed": args.seed,
        "epochs": epochs,
        "device": str(device),
        "protocol_sha256": protocol.sha256,
        "train_seconds": train_secs,
        "calibration_shots": cal_shots,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
