"""DFWM-Hypernetwork：z 生成物理修正网络的权重。

架构：
  z (8维) → HyperNet → 低秩权重偏移 ΔW = A @ B^T
  物理修正 = hidden_state @ (W_base + ΔW) + bias(z)
  最终预测 = WM_base预测 + 物理修正

两阶段训练：
  Stage 1：训练 WM_base + TopologyEncoder（不用 z）
  Stage 2：冻结 WM_base，只训练 HyperNet + ResidualEncoder
  → z 必须被使用，数学上不可能被忽略

先例：HyperNetworks (Ha & Schmidhuber, 2017)
      LoRA (Hu et al., ICLR 2022)
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
CONTEXT_DIM = TOPOLOGY_DIM  # Stage1: WM只用topology（64维）
STATE_DIM = 14
ACTION_DIM = 5
HIDDEN_DIM = 128  # WM GRU hidden size
RANK = 8          # 低秩分解的秩
LAMBDA_A = 5.0    # 物理监督权重


# ── 超网络 ─────────────────────────────────────────────────────────────────────

class PhysicsHyperNet(nn.Module):
    """z → 低秩权重偏移，生成物理修正网络的参数。

    修正量 = hidden_state @ (W_base + A @ B^T) + (b_base + bias_net(z))
    当 z≈0 时退化为零修正（如果 W_base=0，b_base=0 初始化）。
    """

    def __init__(
        self,
        z_dim: int = RESIDUAL_DIM,
        hidden_dim: int = HIDDEN_DIM,
        state_dim: int = STATE_DIM,
        rank: int = RANK,
    ):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # z → 低秩分解矩阵 A (H×r) 和 B (S×r)
        self.A_gen = nn.Sequential(
            nn.Linear(z_dim, 64), nn.SiLU(),
            nn.Linear(64, hidden_dim * rank),
        )
        self.B_gen = nn.Sequential(
            nn.Linear(z_dim, 64), nn.SiLU(),
            nn.Linear(64, state_dim * rank),
        )
        self.bias_gen = nn.Sequential(
            nn.Linear(z_dim, 32), nn.SiLU(),
            nn.Linear(32, state_dim),
        )

        # 可学习的基础权重（初始化为零，保证 z=0 时无修正）
        self.W_base = nn.Parameter(torch.zeros(hidden_dim, state_dim))
        self.b_base = nn.Parameter(torch.zeros(state_dim))

        # 修正幅度缩放（防止训练初期修正过大）
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        z: torch.Tensor,   # (z_dim,) 或 (B, z_dim)
        hidden: torch.Tensor,  # (B, hidden_dim)
    ) -> torch.Tensor:
        """返回状态修正量 (B, state_dim)。"""
        B = hidden.shape[0]
        z_expanded = z.unsqueeze(0).expand(B, -1) if z.dim() == 1 else z

        A = self.A_gen(z_expanded).view(B, self.hidden_dim, self.rank)  # (B, H, r)
        B_mat = self.B_gen(z_expanded).view(B, self.state_dim, self.rank)  # (B, S, r)
        delta_W = torch.bmm(A, B_mat.transpose(-1, -2))  # (B, H, S)

        W = self.W_base.unsqueeze(0) + delta_W  # (B, H, S)
        bias = self.b_base + self.bias_gen(z_expanded)  # (B, S)

        correction = torch.bmm(hidden.unsqueeze(1), W).squeeze(1) + bias  # (B, S)
        return correction * self.scale


# ── Stage 1：训练 WM_base（不用 z）────────────────────────────────────────────

def train_stage1(trajs_by_domain, joint_ranges, *, epochs, device, seed):
    """Stage 1：只用 e_topology，训练 WM_base。"""
    torch.manual_seed(seed)
    topo_enc = TopologyEncoder(TopologyEncoderConfig(out_dim=TOPOLOGY_DIM)).to(device)
    wm_base = WorldModel(WorldModelConfig(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        context_dim=TOPOLOGY_DIM,  # 只用 topology
    )).to(device)
    params = list(topo_enc.parameters()) + list(wm_base.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    items = list(trajs_by_domain.items())

    for epoch in range(epochs):
        losses = []
        np.random.shuffle(items)
        for domain_id, trajs in items:
            if not trajs:
                continue
            topology_name = domain_id.split("__")[0]
            dmg = damage_from_name(topology_name)
            for traj in trajs:
                T = traj.states.shape[0] - 1
                if T < 4:
                    continue
                states = traj.states.to(device)
                actions = traj.actions.to(device)
                e_topo = encode_damage_batch(topo_enc, [dmg], joint_ranges, device)
                loss = rssm_training_loss(wm_base, states.unsqueeze(0), actions.unsqueeze(0), e_topo)
                losses.append(loss)

        total = torch.stack(losses).mean()
        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [S1] epoch {epoch+1}/{epochs}  loss={total.item():.3f}", flush=True)

    return topo_enc, wm_base


# ── Stage 2：训练 HyperNet + Encoder（WM_base 冻结）─────────────────────────────

def multistep_with_hypernetwork(
    wm_base, hypernet, topo_enc,
    states, actions, z, domain_dmg, joint_ranges, device, horizon=10
):
    """用超网络修正的多步 rollout，返回 RMSE。"""
    B = states.shape[0]
    horizon = min(horizon, actions.shape[1])
    sq_errs = []

    e_topo = encode_damage_batch(topo_enc, [domain_dmg] * B, joint_ranges, device)

    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        pred = states[:, start].clone()
        hidden = None
        for h in range(horizon):
            out, hidden = wm_base.step(pred, actions[:, start + h], e_topo, hidden)
            base_pred = out["mean"]
            # 超网络修正：hidden state → 物理修正量
            correction = hypernet(z, hidden)
            pred = base_pred + correction
            target = states[:, start + h + 1]
            sq_errs.append((pred - target).pow(2).mean(dim=-1))

    return torch.stack(sq_errs).mean().sqrt()


def train_stage2(
    trajs_by_domain, joint_ranges,
    topo_enc, wm_base,
    *, epochs, device, seed
):
    """Stage 2：冻结 WM_base，只训练 HyperNet + ResidualEncoder。"""
    torch.manual_seed(seed + 1000)

    # 冻结 Stage 1 的参数
    for p in topo_enc.parameters():
        p.requires_grad_(False)
    for p in wm_base.parameters():
        p.requires_grad_(False)

    hypernet = PhysicsHyperNet().to(device)
    res_enc = ResidualEncoder(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, z_dim=RESIDUAL_DIM,
    ).to(device)

    params = list(hypernet.parameters()) + list(res_enc.parameters())
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

                # Encoder：第一半 → z
                cal_states = states[:split + 1].unsqueeze(0)
                cal_actions = actions[:split].unsqueeze(0)
                z = res_enc(cal_states, cal_actions)

                # WM_base + HyperNet：第二半评估
                states_eval = states[split:].unsqueeze(0)
                actions_eval = actions[split:].unsqueeze(0)
                rmse = multistep_with_hypernetwork(
                    wm_base, hypernet, topo_enc,
                    states_eval, actions_eval, z, dmg, joint_ranges, device,
                )
                wm_losses.append(rmse)

                # 物理监督
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
            print(f"  [S2] epoch {epoch+1}/{epochs}  wm={wm_m:.4f}  phys={ph_m:.4f}", flush=True)

    # 检查 z_norm
    sample = items[0][1][0]
    T = sample.states.shape[0] - 1
    split = T // 2
    with torch.no_grad():
        z_test = res_enc(
            sample.states[:split+1].unsqueeze(0).to(device),
            sample.actions[:split].unsqueeze(0).to(device)
        )
    print(f"  z_norm={z_test.norm().item():.3f}  scale={hypernet.scale.item():.4f}", flush=True)

    return hypernet, res_enc


# ── 评估 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_hypernetwork(
    topo_enc, wm_base, hypernet, res_enc,
    domain, cal_trajs, eval_trajs, joint_ranges, device,
    *, k, horizon=10
):
    topo_enc.eval(); wm_base.eval(); hypernet.eval(); res_enc.eval()
    topology_name = domain.domain_id.split("__")[0]
    physics_name = domain.domain_id.split("__")[1]
    dmg = damage_from_name(topology_name)

    eval_states = torch.stack([t.states for t in eval_trajs]).to(device)
    eval_actions = torch.stack([t.actions for t in eval_trajs]).to(device)

    # 1. 基线（无超网络，z=0）
    e_topo = encode_damage_batch(topo_enc, [dmg] * len(eval_trajs), joint_ranges, device)
    sq_errs_base = []
    horizon_ = min(horizon, eval_actions.shape[1])
    for start in range(0, eval_actions.shape[1] - horizon_ + 1, horizon_):
        pred = eval_states[:, start].clone()
        hidden = None
        for h in range(horizon_):
            out, hidden = wm_base.step(pred, eval_actions[:, start + h], e_topo, hidden)
            pred = out["mean"]
            sq_errs_base.append((pred - eval_states[:, start + h + 1]).pow(2).mean(dim=-1))
    base_rmse = float(torch.stack(sq_errs_base).mean().sqrt())

    # 2. DFWM（超网络 + encoder 推断 z）
    if k == 0 or not cal_trajs:
        z = torch.zeros(RESIDUAL_DIM, device=device)
    else:
        cal = cal_trajs[:k]
        min_T = min(t.states.shape[0] for t in cal)
        cal_states = torch.stack([t.states[:min_T] for t in cal]).to(device)
        cal_actions = torch.stack([t.actions[:min_T-1] for t in cal]).to(device)
        z = res_enc(cal_states, cal_actions)

    dfwm_rmse = float(multistep_with_hypernetwork(
        wm_base, hypernet, topo_enc,
        eval_states, eval_actions, z, dmg, joint_ranges, device, horizon,
    ))

    # 3. Oracle（真实物理 descriptor 作为 z）
    z_oracle = residual_descriptor(physics_name, device=device, dtype=torch.float32)
    oracle_rmse = float(multistep_with_hypernetwork(
        wm_base, hypernet, topo_enc,
        eval_states, eval_actions, z_oracle, dmg, joint_ranges, device, horizon,
    ))

    imp = 100.0 * (base_rmse - dfwm_rmse) / base_rmse
    oracle_imp = 100.0 * (base_rmse - oracle_rmse) / base_rmse
    return {
        "k": k,
        "base_rmse": base_rmse,
        "dfwm_rmse": dfwm_rmse,
        "oracle_rmse": oracle_rmse,
        "dfwm_improvement_pct": imp,
        "oracle_improvement_pct": oracle_imp,
        "z_norm": float(z.norm().item()),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs-s1", type=int, default=20)
    parser.add_argument("--epochs-s2", type=int, default=30)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    steps = args.steps or int(config["steps"])
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)
    cal_shots = list(config["calibration_shots"])
    horizon = int(config["rollout_horizon"])
    output_dir = args.output_dir or (
        Path("runs/g2_dfwm_hypernetwork") / f"seed{args.seed}_v1"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  S1={args.epochs_s1}  S2={args.epochs_s2}  device={device}", flush=True)
    print("架构：WM_base(e_topo) + HyperNet(z) → 物理修正", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    cal_targets = tuple(t.as_array() for t in targets.calibration)
    eval_targets = tuple(t.as_array() for t in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    # 数据采集
    print("\n[data] 采集训练数据...", flush=True)
    t0 = time.perf_counter()
    all_train = collect_push_domains_warp(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=args.seed * 10_000,
        block_initial_xy=block_xy, excitation="active",
    )
    tpd = int(config["trajectories_per_train_domain"])
    trajs_by_domain = {
        domain.domain_id: all_train[i * tpd:(i + 1) * tpd]
        for i, domain in enumerate(protocol.train)
    }
    print(f"  {len(all_train)} 条，{time.perf_counter()-t0:.1f}s", flush=True)

    # Stage 1
    print(f"\n[S1] 训练 WM_base（{args.epochs_s1} epochs）...", flush=True)
    t0 = time.perf_counter()
    topo_enc, wm_base = train_stage1(
        trajs_by_domain, ranges,
        epochs=args.epochs_s1, device=device, seed=args.seed,
    )
    print(f"  完成，{time.perf_counter()-t0:.1f}s", flush=True)

    # Stage 2
    print(f"\n[S2] 训练 HyperNet + Encoder（{args.epochs_s2} epochs，WM冻结）...", flush=True)
    t0 = time.perf_counter()
    hypernet, res_enc = train_stage2(
        trajs_by_domain, ranges, topo_enc, wm_base,
        epochs=args.epochs_s2, device=device, seed=args.seed,
    )
    print(f"  完成，{time.perf_counter()-t0:.1f}s", flush=True)

    # 评估
    rows = []
    for idx, domain in enumerate(protocol.test):
        print(f"\n[eval] {domain.domain_id}...", flush=True)
        cal_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=max(cal_shots),
            steps=steps, seed=args.seed * 100_000 + idx * 1000,
            targets=cal_targets, excitation="active", block_initial_xy=block_xy,
        )
        ev_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps, seed=args.seed * 100_000 + idx * 1000 + 500,
            targets=eval_targets, excitation="goal", block_initial_xy=block_xy,
        )
        for k in cal_shots:
            r = evaluate_hypernetwork(
                topo_enc, wm_base, hypernet, res_enc,
                domain, cal_trajs, ev_trajs, ranges, device,
                k=k, horizon=horizon,
            )
            rows.append({"domain": domain.domain_id, "seed": args.seed, **r})
            print(
                f"  K={k}: base={r['base_rmse']:.4f}  "
                f"dfwm={r['dfwm_rmse']:.4f}({r['dfwm_improvement_pct']:+.2f}%)  "
                f"oracle={r['oracle_rmse']:.4f}({r['oracle_improvement_pct']:+.2f}%)  "
                f"z={r['z_norm']:.3f}",
                flush=True
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"method": "dfwm_hypernetwork", "seed": args.seed,
                    "epochs_s1": args.epochs_s1, "epochs_s2": args.epochs_s2,
                    "rows": rows}, indent=2),
        encoding="utf-8"
    )
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
