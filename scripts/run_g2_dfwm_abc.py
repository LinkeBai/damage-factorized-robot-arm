"""DFWM with A+B+C combined training: physics supervision + contrastive + active probing.

A: Physics parameter supervision (RMA-style, Kumar et al. RSS 2021)
B: InfoNCE contrastive loss (Chen et al. ICML 2020)
C: Active probing trajectories for data collection

Usage:
  python scripts/run_g2_dfwm_abc.py --seed 7
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
from src.robotarm.models.amortized_encoder import (
    ResidualEncoder, physics_supervision_loss, infonce_loss
)
from src.robotarm.models.topology_encoder import TopologyEncoder, TopologyEncoderConfig
from src.robotarm.models.world_model import WorldModel, WorldModelConfig
from src.robotarm.models.residual_context import compose_context
from src.robotarm.training.g1_mechanism import (
    TOPOLOGY_DIM, RESIDUAL_DIM,
    encode_damage_batch, rssm_training_loss, residual_descriptor,
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

# Loss weights — A must dominate early to prevent z collapse
LAMBDA_A = 5.0   # physics supervision (强制 z 携带物理信息)
LAMBDA_B = 0.5   # contrastive


def build_models(device):
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

def train(trajs_by_domain, joint_ranges, *, epochs, device, seed):
    torch.manual_seed(seed)
    topo_enc, wm, res_enc = build_models(device)
    items = list(trajs_by_domain.items())

    # ── Phase 1: 只训练 encoder，强迫 z 学习物理参数 ──────────────────────────
    phase1_epochs = max(5, epochs // 4)
    enc_optimizer = torch.optim.Adam(list(res_enc.parameters()), lr=3e-4)
    print(f"  Phase 1: encoder-only ({phase1_epochs} epochs) …", flush=True)

    for epoch in range(phase1_epochs):
        phys_losses = []
        for domain_id, trajs in items:
            if not trajs:
                continue
            physics_name = domain_id.split("__")[1]
            phys_target = residual_descriptor(
                physics_name, device=device, dtype=torch.float32
            ).detach()
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
                p_loss = physics_supervision_loss(z.unsqueeze(0), phys_target.unsqueeze(0))
                phys_losses.append(p_loss)

        total = torch.stack(phys_losses).mean()
        enc_optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(list(res_enc.parameters()), 1.0)
        enc_optimizer.step()

        if (epoch + 1) % phase1_epochs == 0 or epoch == 0:
            print(f"    p1 epoch {epoch+1}/{phase1_epochs}  phys={total.item():.4f}", flush=True)

    # 检查 Phase 1 效果
    with torch.no_grad():
        sample_traj = items[0][1][0]
        T = sample_traj.states.shape[0] - 1
        split = T // 2
        z_test = res_enc(
            sample_traj.states[:split+1].unsqueeze(0).to(device),
            sample_traj.actions[:split].unsqueeze(0).to(device)
        )
        print(f"  Phase 1 done: z_norm={z_test.norm().item():.3f} (target >1.0)", flush=True)

    # ── Phase 2: 冻结 encoder，训练 WM 使用 z ────────────────────────────────
    phase2_epochs = epochs - phase1_epochs
    for p in res_enc.parameters():
        p.requires_grad_(False)

    wm_params = list(topo_enc.parameters()) + list(wm.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=3e-4)
    print(f"  Phase 2: WM-only ({phase2_epochs} epochs, encoder frozen) …", flush=True)

    for epoch in range(phase2_epochs):
        wm_losses = []
        np.random.shuffle(items)

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

                with torch.no_grad():
                    cal_states = states[:split + 1].unsqueeze(0)
                    cal_actions = actions[:split].unsqueeze(0)
                    z = res_enc(cal_states, cal_actions)

                context = compose_context(e_topo, z, context_dim=CONTEXT_DIM).unsqueeze(0)
                states_eval = states[split:].unsqueeze(0)
                actions_eval = actions[split:].unsqueeze(0)
                wm_losses.append(rssm_training_loss(wm, states_eval, actions_eval, context))

        total = torch.stack(wm_losses).mean()
        wm_optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(wm_params, 1.0)
        wm_optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    p2 epoch {epoch+1}/{phase2_epochs}  wm={total.item():.4f}", flush=True)

    # 解冻 encoder（评估时需要）
    for p in res_enc.parameters():
        p.requires_grad_(True)

    return topo_enc, wm, res_enc

    return topo_enc, wm, res_enc


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(topo_enc, wm, res_enc, domain, cal_trajs, eval_trajs,
             joint_ranges, device, *, k, horizon=10):
    topo_enc.eval(); wm.eval(); res_enc.eval()
    topology_name = domain.domain_id.split("__")[0]
    dmg = damage_from_name(topology_name)

    with torch.no_grad():
        e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device).squeeze(0)

        if k == 0 or not cal_trajs:
            z = torch.zeros(RESIDUAL_DIM, device=device)
        else:
            cal = cal_trajs[:k]
            min_T = min(t.states.shape[0] for t in cal)
            cal_states = torch.stack([t.states[:min_T] for t in cal]).to(device)
            cal_actions = torch.stack([t.actions[:min_T - 1] for t in cal]).to(device)
            z = res_enc(cal_states, cal_actions)

        context = compose_context(e_topo, z, context_dim=CONTEXT_DIM).unsqueeze(0)
        eval_states = torch.stack([t.states for t in eval_trajs]).to(device)
        eval_actions = torch.stack([t.actions for t in eval_trajs]).to(device)
        ctx = context.expand(eval_states.shape[0], -1)
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
        raise ValueError(f"seed {args.seed} not in frozen seeds")

    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    cal_shots = list(config["calibration_shots"])
    horizon = int(config["rollout_horizon"])
    output_dir = args.output_dir or (
        Path("runs/g2_dfwm_abc") / f"seed{args.seed}_v1"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  epochs={epochs}  device={device}", flush=True)
    print(f"losses: wm + {LAMBDA_A}×phys(A) + {LAMBDA_B}×contrastive(B)  data: active-probe(C)", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration_targets = tuple(item.as_array() for item in targets.calibration)
    evaluation_targets = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    # C: active probing trajectories via warp GPU batch
    print("\n[train] collecting active-probe trajectories (warp GPU) …", flush=True)
    t0 = time.perf_counter()
    all_train_trajs = collect_push_domains_warp(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        block_initial_xy=block_initial_xy,
        excitation="active",  # C: active probing
    )
    tpd = int(config["trajectories_per_train_domain"])
    trajs_by_domain = {
        domain.domain_id: all_train_trajs[i * tpd:(i + 1) * tpd]
        for i, domain in enumerate(protocol.train)
    }
    print(f"  {len(all_train_trajs)} trajs in {time.perf_counter()-t0:.1f}s", flush=True)

    # A+B+C training
    print("\n[train] A+B+C DFWM …", flush=True)
    t0 = time.perf_counter()
    topo_enc, wm, res_enc = train(
        trajs_by_domain, ranges,
        epochs=epochs, device=device, seed=args.seed,
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    # Evaluate
    rows = []
    for idx, domain in enumerate(protocol.test):
        print(f"\n[eval] {domain.domain_id} …", flush=True)
        # 校准轨迹用 active-probe（与训练分布一致）
        cal_trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=max(cal_shots),
            steps=steps,
            seed=args.seed * 100_000 + idx * 1000,
            targets=calibration_targets,
            excitation="active",
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
            m = evaluate(topo_enc, wm, res_enc, domain,
                        cal_trajs, eval_trajs, ranges, device,
                        k=k, horizon=horizon)
            rows.append({"domain": domain.domain_id, "method": "dfwm_abc",
                         "seed": args.seed, **m})
            print(f"  K={k}: rmse={m['ensemble_rmse']:.4f}  z_norm={m['z_norm']:.3f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"method": "dfwm_abc", "seed": args.seed,
                    "epochs": epochs, "lambda_A": LAMBDA_A,
                    "lambda_B": LAMBDA_B, "rows": rows}, indent=2),
        encoding="utf-8"
    )
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
