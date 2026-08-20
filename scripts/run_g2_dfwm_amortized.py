"""DFWM with amortized residual encoder (Plan V6 §4.3B).

KEY DIFFERENCE from latent optimization:
  - Latent opt: per-deployment Adam loop on K short trajs -> overfits, doesn't generalize
  - Amortized encoder: trained jointly with WM across many (topology, physics) pairs
    -> learns generalizable mapping: trajectory patterns -> z_residual
    -> at test time: forward pass only, no optimization needed

Training protocol:
  For each training trajectory:
    1. Split into first-half (calibration) and second-half (evaluation)
    2. encoder(first-half states, actions) -> z_residual
    3. WM([e_topology, z_residual], second-half) -> prediction loss
    4. Backprop through BOTH encoder and WM jointly

Test time (K-shot):
  K=0: z = zeros
  K>=1: encoder(K calibration trajectories) -> z -> predict eval trajectories

Usage:
  python scripts/run_g2_dfwm_amortized.py --seed 7
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
from src.robotarm.models.amortized_encoder import ResidualEncoder
from src.robotarm.models.topology_encoder import TopologyEncoder, TopologyEncoderConfig
from src.robotarm.models.world_model import WorldModel, WorldModelConfig
from src.robotarm.models.residual_context import compose_context
from src.robotarm.training.g1_mechanism import (
    TOPOLOGY_DIM, RESIDUAL_DIM,
    encode_damage_batch,
    rssm_training_loss,
)
from src.robotarm.training.sim_protocol import load_g1_protocol, damage_from_name
from src.robotarm.training.target_split import load_target_split
from src.robotarm.training.sim_data import SimTrajectory
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_dfwm_v1.yaml")
CONTEXT_DIM = TOPOLOGY_DIM + RESIDUAL_DIM  # 72
STATE_DIM = 14
ACTION_DIM = 5


def build_models(device: torch.device):
    topo_enc = TopologyEncoder(TopologyEncoderConfig(out_dim=TOPOLOGY_DIM)).to(device)
    wm = WorldModel(WorldModelConfig(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, context_dim=CONTEXT_DIM
    )).to(device)
    res_enc = ResidualEncoder(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=128, z_dim=RESIDUAL_DIM,
    ).to(device)
    return topo_enc, wm, res_enc


def multistep_rmse(wm, states, actions, context, horizon=10):
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

def train(
    trajs_by_domain: dict[str, list[SimTrajectory]],
    joint_ranges: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    topo_enc, wm, res_enc = build_models(device)
    params = (
        list(topo_enc.parameters()) +
        list(wm.parameters()) +
        list(res_enc.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=3e-4)

    items = list(trajs_by_domain.items())

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        np.random.shuffle(items)

        # Batch all domains together for one gradient step per epoch
        all_losses = []
        for domain_id, trajs in items:
            if not trajs:
                continue
            topology_name = domain_id.split("__")[0]
            dmg = damage_from_name(topology_name)
            e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device).squeeze(0)

            for traj in trajs:
                T = traj.states.shape[0] - 1
                if T < 6:
                    continue
                split = T // 2
                states = traj.states.to(device)
                actions = traj.actions.to(device)

                cal_states = states[:split + 1].unsqueeze(0)
                cal_actions = actions[:split].unsqueeze(0)
                z = res_enc(cal_states, cal_actions)

                context = compose_context(e_topo, z, context_dim=CONTEXT_DIM).unsqueeze(0)
                states_eval = states[split:].unsqueeze(0)
                actions_eval = actions[split:].unsqueeze(0)

                loss = rssm_training_loss(wm, states_eval, actions_eval, context)
                all_losses.append(loss)
                n_batches += 1

        # Single backward over all domain losses
        if all_losses:
            total_loss = torch.stack(all_losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss = total_loss.detach().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}", flush=True)

    return topo_enc, wm, res_enc


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    topo_enc, wm, res_enc,
    domain,
    calibration_trajs: list[SimTrajectory],
    evaluation_trajs: list[SimTrajectory],
    joint_ranges: np.ndarray,
    device: torch.device,
    *,
    k: int,
    horizon: int = 10,
) -> dict:
    topo_enc.eval(); wm.eval(); res_enc.eval()

    topology_name = domain.domain_id.split("__")[0]
    dmg = damage_from_name(topology_name)

    with torch.no_grad():
        e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device).squeeze(0)

        if k == 0 or not calibration_trajs:
            z = torch.zeros(RESIDUAL_DIM, device=device)
        else:
            cal = calibration_trajs[:k]
            # Pad to same length if needed
            min_T = min(t.states.shape[0] for t in cal)
            cal_states = torch.stack([t.states[:min_T] for t in cal]).to(device)   # (K, T, 14)
            cal_actions = torch.stack([t.actions[:min_T-1] for t in cal]).to(device)  # (K, T-1, 5)
            z = res_enc(cal_states, cal_actions)

        context = compose_context(e_topo, z, context_dim=CONTEXT_DIM).unsqueeze(0)

        eval_states = torch.stack([t.states for t in evaluation_trajs]).to(device)
        eval_actions = torch.stack([t.actions for t in evaluation_trajs]).to(device)
        B = eval_states.shape[0]
        ctx = context.expand(B, -1)

        rmse = multistep_rmse(wm, eval_states, eval_actions, ctx, horizon)

    return {"k": k, "ensemble_rmse": rmse, "z_norm": float(z.norm().item())}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
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
        Path("runs/g2_dfwm_amortized") / f"seed{args.seed}_v1"
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

    # ── collect training data via warp (GPU batch, random excitation) ─────────
    print("\n[train] collecting trajectories (warp GPU batch) …", flush=True)
    t0 = time.perf_counter()
    all_train_trajs = collect_push_domains_warp(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        block_initial_xy=block_initial_xy,
    )
    tpd = int(config["trajectories_per_train_domain"])
    trajs_by_domain: dict[str, list[SimTrajectory]] = {}
    for i, domain in enumerate(protocol.train):
        trajs_by_domain[domain.domain_id] = all_train_trajs[i * tpd:(i + 1) * tpd]
    print(f"  {len(all_train_trajs)} trajs in {time.perf_counter()-t0:.1f}s", flush=True)

    # ── train jointly ─────────────────────────────────────────────────────────
    print("\n[train] DFWM amortized encoder …", flush=True)
    t0 = time.perf_counter()
    topo_enc, wm, res_enc = train(
        trajs_by_domain, ranges,
        epochs=epochs, device=device, seed=args.seed,
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    rows = []
    for idx, domain in enumerate(protocol.test):
        print(f"\n[eval] {domain.domain_id} …", flush=True)

        # goal-directed calibration trajectories
        cal_trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=max(cal_shots),
            steps=steps,
            seed=args.seed * 100_000 + idx * 1000,
            targets=calibration_targets,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )

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
            metrics = evaluate(
                topo_enc, wm, res_enc, domain,
                cal_trajs, eval_trajs, ranges, device,
                k=k, horizon=horizon,
            )
            row = {"domain": domain.domain_id, "method": "dfwm_amortized",
                   "seed": args.seed, **metrics}
            rows.append(row)
            print(f"  K={k}: rmse={metrics['ensemble_rmse']:.4f}  z_norm={metrics['z_norm']:.3f}", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(
        json.dumps({"method": "dfwm_amortized", "seed": args.seed,
                    "epochs": epochs, "rows": rows}, indent=2),
        encoding="utf-8"
    )
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
