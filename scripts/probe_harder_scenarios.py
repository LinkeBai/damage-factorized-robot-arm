"""快速检验：固定探测协议在更难场景下的识别准确率。

测试两个更难场景：
1. D4 故障（关节4锁定，训练集从未见过）
2. mixed_unseen 物理参数（比 mixed_composition 更极端）

如果准确率下降 → 最优探测有实际价值
如果仍然100% → 直接写论文

Usage:
  python scripts/probe_harder_scenarios.py --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.robotarm.envs.mujoco_env import MujocoArmEnv
from src.robotarm.training.topology_ensemble import (
    train_topology_ensemble, encode_damage_batch, conditioning_damages,
)
from src.robotarm.training.sim_protocol import load_g1_protocol, damage_from_name, DomainSpec
from src.robotarm.training.target_split import load_target_split
from src.robotarm.envs.residual_physics import residual_profile
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_push_ensemble_v1.yaml")
STATE_NAMES = [
    "q1","q2","q3","q4","q5",
    "dq1","dq2","dq3","dq4","dq5",
    "block_x","block_y","block_vx","block_vy"
]
N_PROBE = 10  # 每个 domain 收集多少条探测轨迹


@torch.no_grad()
def compute_fingerprint(ensemble, domain, trajectories, joint_ranges, device, horizon=10):
    states = torch.stack([t.states for t in trajectories]).to(device)
    actions = torch.stack([t.actions for t in trajectories]).to(device)
    damages = conditioning_damages([domain.damage] * len(trajectories), "constant")
    contexts = [encode_damage_batch(m.encoder, damages, joint_ranges, device) for m in ensemble]
    horizon = min(horizon, actions.shape[1])
    per_dim_vars = []
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        preds = [states[:, start].clone() for _ in ensemble]
        hidden = [None] * len(ensemble)
        for offset in range(horizon):
            means = []
            for i, member in enumerate(ensemble):
                out, hidden[i] = member.world_model.step(
                    preds[i], actions[:, start + offset], contexts[i], hidden[i]
                )
                preds[i] = out["mean"]
                means.append(out["mean"])
            stacked = torch.stack(means)
            per_dim_vars.append(stacked.var(dim=0, unbiased=False).mean(dim=0).cpu().numpy())
    return np.stack(per_dim_vars).mean(axis=0)


def train_classifier(fingerprints_dict, device):
    """训练指纹分类器，返回分类函数。"""
    X, y, labels = [], [], []
    label_map = {}
    for domain_id, fps in fingerprints_dict.items():
        topo = domain_id.split("__")[0]
        if topo not in label_map:
            label_map[topo] = len(label_map)
        for fp in fps:
            X.append(fp)
            y.append(label_map[topo])
            labels.append(topo)

    X_t = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    mean = X_t.mean(0, keepdim=True)
    std = X_t.std(0, keepdim=True).clamp(1e-6)
    X_norm = (X_t - mean) / std

    inv_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    from torch import nn
    clf = nn.Sequential(
        nn.Linear(14, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, n_classes)
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(300):
        loss = F.cross_entropy(clf(X_norm), y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        acc = (clf(X_norm).argmax(1) == y_t).float().mean().item()

    def predict(fp):
        x = torch.tensor((fp - mean.cpu().numpy()) / std.cpu().numpy().clip(1e-6),
                          dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return inv_map[int(clf(x).argmax(dim=-1).item())]

    return predict, acc, label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    steps = int(config["steps"])
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  device={device}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    cal_targets = tuple(t.as_array() for t in targets.calibration)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    # 训练普通集成
    print("\n[1] 训练普通集成 ...", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=args.seed * 10_000,
        targets=cal_targets, excitation="goal", block_initial_xy=block_xy,
    )
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=3,
        epochs=int(config["epochs"]),
        device=device, seed=args.seed, condition_mode="constant",
    )

    # 收集训练 domain 的指纹（用于训练分类器）
    print("\n[2] 收集训练 domain 指纹 ...", flush=True)
    train_fps = {}
    for i, domain in enumerate(protocol.train):
        trajs = collect_push_domains(
            (domain,), trajectories_per_domain=N_PROBE,
            steps=steps, seed=args.seed * 200_000 + i * 1000,
            targets=cal_targets, excitation="active", block_initial_xy=block_xy,
        )
        fps = [compute_fingerprint(ensemble, domain, [t], ranges, device) for t in trajs]
        train_fps[domain.domain_id] = fps

    predict, train_acc, label_map = train_classifier(train_fps, device)
    print(f"  分类器训练准确率: {train_acc:.1%}  类别: {list(label_map.keys())}", flush=True)

    # ── 测试1：原始测试 domain（baseline）──────────────────────────────────────
    print("\n[3] 测试原始场景（D2/D3 mixed_composition）...", flush=True)
    for i, domain in enumerate(protocol.test):
        trajs = collect_push_domains(
            (domain,), trajectories_per_domain=N_PROBE,
            steps=steps, seed=args.seed * 300_000 + i * 1000,
            targets=cal_targets, excitation="active", block_initial_xy=block_xy,
        )
        fps = [compute_fingerprint(ensemble, domain, [t], ranges, device) for t in trajs]
        correct = sum(predict(fp) == domain.domain_id.split("__")[0] for fp in fps)
        print(f"  {domain.domain_id}: {correct}/{N_PROBE} = {correct/N_PROBE:.0%}", flush=True)

    # ── 测试2：D4 故障（从未训练过）──────────────────────────────────────────
    print("\n[4] 测试 D4（训练集从未见过的故障）...", flush=True)
    from src.robotarm.envs.damage import D4
    d4_domain = DomainSpec(topology="D4", residual_name="mixed_composition", split="test")
    try:
        trajs = collect_push_domains(
            (d4_domain,), trajectories_per_domain=N_PROBE,
            steps=steps, seed=args.seed * 400_000,
            targets=cal_targets, excitation="active", block_initial_xy=block_xy,
        )
        fps = [compute_fingerprint(ensemble, d4_domain, [t], ranges, device) for t in trajs]
        preds = [predict(fp) for fp in fps]
        print(f"  D4__mixed_composition 预测分布: {dict(zip(*np.unique(preds, return_counts=True)))}", flush=True)
        print(f"  （D4 不在训练类别中，预测为哪个类？）", flush=True)
        # 检查 D4 指纹和 D2/D3 的区分度
        d4_fp = np.mean(fps, axis=0)
        for train_id, train_fps_list in train_fps.items():
            topo = train_id.split("__")[0]
            other_fp = np.mean(train_fps_list, axis=0)
            cos = np.dot(d4_fp, other_fp) / (np.linalg.norm(d4_fp) * np.linalg.norm(other_fp))
            print(f"  D4 vs {topo:<8}: cosine={cos:.4f}", flush=True)
    except Exception as e:
        print(f"  D4 测试失败: {e}", flush=True)

    # ── 测试3：mixed_unseen 物理（更极端）─────────────────────────────────────
    print("\n[5] 测试 mixed_unseen 物理（比 mixed_composition 更极端）...", flush=True)
    for topo_name in ["D2", "D3"]:
        domain = DomainSpec(topology=topo_name, residual_name="mixed_unseen", split="test")
        try:
            trajs = collect_push_domains(
                (domain,), trajectories_per_domain=N_PROBE,
                steps=steps, seed=args.seed * 500_000,
                targets=cal_targets, excitation="active", block_initial_xy=block_xy,
            )
            fps = [compute_fingerprint(ensemble, domain, [t], ranges, device) for t in trajs]
            correct = sum(predict(fp) == topo_name for fp in fps)
            print(f"  {topo_name}__mixed_unseen: {correct}/{N_PROBE} = {correct/N_PROBE:.0%}", flush=True)
        except Exception as e:
            print(f"  {topo_name}__mixed_unseen 失败: {e}", flush=True)

    print("\n" + "="*50, flush=True)
    print("判断标准：", flush=True)
    print("  D2/D3 mixed_composition < 100% → 现有方法已有瓶颈", flush=True)
    print("  D2/D3 mixed_unseen < 100%      → 最优探测有实际价值", flush=True)
    print("  D4 预测集中在某一类            → 可扩展到新故障类型", flush=True)


if __name__ == "__main__":
    main()
