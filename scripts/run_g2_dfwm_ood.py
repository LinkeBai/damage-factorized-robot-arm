"""DFWM-OOD + Topology Dropout：两个问题同时解决。

OOD split：训练只见 nominal+weak_motor → encoder 有强信号学物理（phys loss 0.0065）
Topology Dropout：训练时 50% 遮蔽 e_topology → WM 必须依赖 z 做预测

这是第一次两个问题都有解的组合。

Usage:
  python scripts/run_g2_dfwm_ood.py --seed 7
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
from src.robotarm.models.amortized_encoder import ResidualEncoder, physics_supervision_loss
from src.robotarm.models.topology_encoder import TopologyEncoder, TopologyEncoderConfig
from src.robotarm.models.world_model import WorldModel, WorldModelConfig
from src.robotarm.models.residual_context import compose_context
from src.robotarm.training.g1_mechanism import (
    TOPOLOGY_DIM, RESIDUAL_DIM,
    encode_damage_batch, rssm_training_loss, residual_descriptor,
)
from src.robotarm.training.sim_protocol import load_g1_protocol, damage_from_name
from src.robotarm.training.target_split import load_target_split
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_dfwm_ood_v1.yaml")
CONTEXT_DIM = TOPOLOGY_DIM + RESIDUAL_DIM  # 72
STATE_DIM = 14
ACTION_DIM = 5
LAMBDA_A = 5.0
TOPO_DROPOUT_P = 0.5  # 遮蔽 e_topology 的概率


def build_models(device):
    topo_enc = TopologyEncoder(TopologyEncoderConfig(out_dim=TOPOLOGY_DIM)).to(device)
    wm = WorldModel(WorldModelConfig(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, context_dim=CONTEXT_DIM
    )).to(device)
    res_enc = ResidualEncoder(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, z_dim=RESIDUAL_DIM,  # 更大的 encoder
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


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(trajs_by_domain, joint_ranges, *, epochs, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    topo_enc, wm, res_enc = build_models(device)
    params = (list(topo_enc.parameters()) +
              list(wm.parameters()) +
              list(res_enc.parameters()))
    optimizer = torch.optim.Adam(params, lr=3e-4)
    items = list(trajs_by_domain.items())

    for epoch in range(epochs):
        wm_losses, phys_losses = [], []
        np.random.shuffle(items)

        for domain_id, trajs in items:
            if not trajs:
                continue
            topology_name = domain_id.split("__")[0]
            physics_name = domain_id.split("__")[1]
            dmg = damage_from_name(topology_name)
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

                e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device).squeeze(0)
                # Topology Dropout：50% 遮蔽 e_topology，强迫 WM 依赖 z
                if rng.random() < TOPO_DROPOUT_P:
                    e_topo_input = torch.zeros_like(e_topo)
                else:
                    e_topo_input = e_topo
                context = compose_context(e_topo_input, z, context_dim=CONTEXT_DIM).unsqueeze(0)

                states_eval = states[split:].unsqueeze(0)
                actions_eval = actions[split:].unsqueeze(0)
                wm_loss = rssm_training_loss(wm, states_eval, actions_eval, context)
                wm_losses.append(wm_loss)

                phys_loss = physics_supervision_loss(z.unsqueeze(0), phys_target.unsqueeze(0))
                phys_losses.append(phys_loss)

        total = torch.stack(wm_losses).mean() + LAMBDA_A * torch.stack(phys_losses).mean()
        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            wm_m = torch.stack(wm_losses).mean().item()
            ph_m = torch.stack(phys_losses).mean().item()
            print(f"  epoch {epoch+1}/{epochs}  wm={wm_m:.3f}  phys={ph_m:.4f}", flush=True)

    # 检查 z_norm
    sample = items[0][1][0]
    T = sample.states.shape[0] - 1
    split = T // 2
    with torch.no_grad():
        z_test = res_enc(
            sample.states[:split+1].unsqueeze(0).to(device),
            sample.actions[:split].unsqueeze(0).to(device)
        )
    print(f"  z_norm={z_test.norm().item():.3f}", flush=True)
    return topo_enc, wm, res_enc


# ── 评估（三种方法）────────────────────────────────────────────────────────────

def evaluate_all(topo_enc, wm, res_enc, domain, cal_trajs, eval_trajs,
                 joint_ranges, device, *, k, horizon=10):
    topo_enc.eval(); wm.eval(); res_enc.eval()
    topology_name = domain.domain_id.split("__")[0]
    physics_name = domain.domain_id.split("__")[1]
    dmg = damage_from_name(topology_name)

    eval_states = torch.stack([t.states for t in eval_trajs]).to(device)
    eval_actions = torch.stack([t.actions for t in eval_trajs]).to(device)

    with torch.no_grad():
        e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device).squeeze(0)

        results = {}

        # 1. 普通集成基线（z=0）
        z_zero = torch.zeros(RESIDUAL_DIM, device=device)
        ctx_base = compose_context(e_topo, z_zero, context_dim=CONTEXT_DIM).unsqueeze(0)
        ctx_base = ctx_base.expand(eval_states.shape[0], -1)
        results["ordinary_k0"] = multistep_rmse(wm, eval_states, eval_actions, ctx_base, horizon)

        # 2. DFWM amortized（encoder推断z）
        if k > 0 and cal_trajs:
            cal = cal_trajs[:k]
            min_T = min(t.states.shape[0] for t in cal)
            cal_states = torch.stack([t.states[:min_T] for t in cal]).to(device)
            cal_actions = torch.stack([t.actions[:min_T-1] for t in cal]).to(device)
            z_pred = res_enc(cal_states, cal_actions)
        else:
            z_pred = z_zero

        ctx_dfwm = compose_context(e_topo, z_pred, context_dim=CONTEXT_DIM).unsqueeze(0)
        ctx_dfwm = ctx_dfwm.expand(eval_states.shape[0], -1)
        results[f"dfwm_k{k}"] = multistep_rmse(wm, eval_states, eval_actions, ctx_dfwm, horizon)

        # 3. Oracle：直接用真实物理 descriptor 作为 z（上界）
        z_oracle = residual_descriptor(physics_name, device=device, dtype=torch.float32)
        ctx_oracle = compose_context(e_topo, z_oracle, context_dim=CONTEXT_DIM).unsqueeze(0)
        ctx_oracle = ctx_oracle.expand(eval_states.shape[0], -1)
        results["oracle"] = multistep_rmse(wm, eval_states, eval_actions, ctx_oracle, horizon)

        results["z_norm_pred"] = float(z_pred.norm().item())
        results["z_norm_oracle"] = float(z_oracle.norm().item())

    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

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
        Path("runs/g2_dfwm_ood") / f"seed{args.seed}_v1"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  epochs={epochs}  device={device}", flush=True)
    print("训练物理：nominal + weak_motor | 测试物理：high_damping + delay_1（OOD）", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration_targets = tuple(item.as_array() for item in targets.calibration)
    evaluation_targets = tuple(item.as_array() for item in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    # 训练数据（warp GPU 批量）
    print("\n[train] 采集训练数据...", flush=True)
    t0 = time.perf_counter()
    all_train_trajs = collect_push_domains_warp(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=args.seed * 10_000,
        block_initial_xy=block_initial_xy, excitation="active",
    )
    tpd = int(config["trajectories_per_train_domain"])
    trajs_by_domain = {
        domain.domain_id: all_train_trajs[i * tpd:(i + 1) * tpd]
        for i, domain in enumerate(protocol.train)
    }
    print(f"  {len(all_train_trajs)} 条轨迹，{time.perf_counter()-t0:.1f}s", flush=True)

    # 训练
    print("\n[train] DFWM-OOD...", flush=True)
    t0 = time.perf_counter()
    topo_enc, wm, res_enc = train(
        trajs_by_domain, ranges,
        epochs=epochs, device=device, seed=args.seed,
    )
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

    # 评估
    rows = []
    for idx, domain in enumerate(protocol.test):
        print(f"\n[eval] {domain.domain_id}（OOD 物理）...", flush=True)

        cal_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=max(cal_shots),
            steps=steps, seed=args.seed * 100_000 + idx * 1000,
            targets=calibration_targets, excitation="active",
            block_initial_xy=block_initial_xy,
        )
        eval_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps, seed=args.seed * 100_000 + idx * 1000 + 500,
            targets=evaluation_targets, excitation="goal",
            block_initial_xy=block_initial_xy,
        )

        for k in cal_shots:
            r = evaluate_all(topo_enc, wm, res_enc, domain,
                           cal_trajs, eval_trajs, ranges, device,
                           k=k, horizon=horizon)
            base = r["ordinary_k0"]
            dfwm = r[f"dfwm_k{k}"]
            oracle = r["oracle"]
            imp = 100.0 * (base - dfwm) / base
            oracle_imp = 100.0 * (base - oracle) / base
            rows.append({
                "domain": domain.domain_id, "seed": args.seed, "k": k,
                "ordinary_rmse": base, "dfwm_rmse": dfwm, "oracle_rmse": oracle,
                "dfwm_improvement_pct": imp, "oracle_improvement_pct": oracle_imp,
                "z_norm_pred": r["z_norm_pred"], "z_norm_oracle": r["z_norm_oracle"],
            })
            print(f"  K={k}: ordinary={base:.4f}  dfwm={dfwm:.4f}({imp:+.2f}%)  "
                  f"oracle={oracle:.4f}({oracle_imp:+.2f}%)  z_norm={r['z_norm_pred']:.3f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"method": "dfwm_ood", "seed": args.seed,
                    "epochs": epochs, "rows": rows}, indent=2),
        encoding="utf-8"
    )
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
