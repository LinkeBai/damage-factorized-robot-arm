"""可行性探针：集成分歧的空间模式是否能区分 D2 vs D3。

假设：D2 和 D3 锁定不同关节，导致集成成员在不同状态维度上
的分歧有不同的空间分布。如果这个假设成立，可以用分歧指纹
识别故障拓扑，无需显式 topology descriptor。

输出：
  - 每个 domain 的平均分歧指纹（14维，按状态维度）
  - D2 vs D3 指纹的 cosine 相似度（越低越好）
  - 线性可分性：用 z-score 分类准确率估计

Usage:
  python scripts/probe_disagreement_fingerprint.py --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.robotarm.envs.mujoco_env import MujocoArmEnv
from src.robotarm.training.topology_ensemble import (
    train_topology_ensemble, evaluate_topology_ensemble,
    encode_damage_batch,
)
from src.robotarm.training.sim_protocol import load_g1_protocol
from src.robotarm.training.target_split import load_target_split
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_push_ensemble_v1.yaml")
STATE_NAMES = [
    "q1","q2","q3","q4","q5",
    "dq1","dq2","dq3","dq4","dq5",
    "block_x","block_y","block_vx","block_vy"
]


@torch.no_grad()
def compute_disagreement_fingerprint(
    ensemble, domain, trajectories, joint_ranges, device, horizon=10
):
    """返回 per-state-dim 分歧指纹 (14,)。"""
    from src.robotarm.training.topology_ensemble import conditioning_damages
    from src.robotarm.training.sim_protocol import damage_from_name

    states = torch.stack([t.states for t in trajectories]).to(device)
    actions = torch.stack([t.actions for t in trajectories]).to(device)

    damages = conditioning_damages(
        [domain.damage] * len(trajectories), "constant"
    )
    contexts = [
        encode_damage_batch(m.encoder, damages, joint_ranges, device)
        for m in ensemble
    ]

    horizon = min(horizon, actions.shape[1])
    per_dim_var = []  # (state_dim,) per rollout window

    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        predictions = [states[:, start].clone() for _ in ensemble]
        hidden = [None for _ in ensemble]
        for offset in range(horizon):
            means = []
            for i, member in enumerate(ensemble):
                out, hidden[i] = member.world_model.step(
                    predictions[i], actions[:, start + offset],
                    contexts[i], hidden[i]
                )
                predictions[i] = out["mean"]
                means.append(out["mean"])
            stacked = torch.stack(means)  # (M, B, state_dim)
            # Per-dim variance: mean over batch
            var = stacked.var(dim=0, unbiased=False).mean(dim=0)  # (state_dim,)
            per_dim_var.append(var.cpu().numpy())

    fingerprint = np.stack(per_dim_var).mean(axis=0)  # (state_dim,)
    return fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    epochs = int(config["epochs"])
    steps = int(config["steps"])
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  device={device}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(t.as_array() for t in targets.calibration)
    evaluation = tuple(t.as_array() for t in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("[train] collecting …", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=args.seed * 10_000,
        targets=calibration, excitation="goal", block_initial_xy=block_xy,
    )

    print("[train] ordinary ensemble …", flush=True)
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=3, epochs=epochs,
        device=device, seed=args.seed, condition_mode="constant",
    )

    print("[probe] computing disagreement fingerprints …", flush=True)
    fingerprints = {}
    for domain in protocol.test:
        test_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=5,
            steps=steps, seed=args.seed * 100_000,
            targets=evaluation, excitation="goal", block_initial_xy=block_xy,
        )
        fp = compute_disagreement_fingerprint(
            ensemble, domain, test_trajs, ranges, device
        )
        fingerprints[domain.domain_id] = fp
        print(f"  {domain.domain_id}: mean_disagree={fp.mean():.5f}")

    # 分析
    domains = list(fingerprints.keys())
    print("\n=== Per-Dimension Disagreement Fingerprint ===")
    print(f"{'Dim':>4}  {'State':>8}", end="")
    for d in domains:
        print(f"  {d[:20]:>20}", end="")
    print()
    for i, name in enumerate(STATE_NAMES):
        print(f"{i:>4}  {name:>8}", end="")
        for d in domains:
            print(f"  {fingerprints[d][i]:>20.5f}", end="")
        print()

    if len(domains) == 2:
        d0, d1 = domains
        fp0, fp1 = fingerprints[d0], fingerprints[d1]

        # Cosine similarity
        cos = np.dot(fp0, fp1) / (np.linalg.norm(fp0) * np.linalg.norm(fp1))
        print(f"\nCosine similarity D2 vs D3: {cos:.4f}  (↓ 越低 → 越可区分)")

        # 差异最大的维度
        diff = np.abs(fp0 - fp1)
        top5 = np.argsort(diff)[::-1][:5]
        print("\n最大差异的5个状态维度：")
        for idx in top5:
            print(f"  {STATE_NAMES[idx]:>8}: D2={fp0[idx]:.5f}  D3={fp1[idx]:.5f}  diff={diff[idx]:.5f}")

        # 简单线性可分性：按最大差异维度二分类
        best_dim = top5[0]
        threshold = (fp0[best_dim] + fp1[best_dim]) / 2
        print(f"\n单维度分类（{STATE_NAMES[best_dim]}，threshold={threshold:.5f}）：")
        print(f"  D2预测为{'D2' if fp0[best_dim] > threshold else 'D3'}  ← {'正确' if fp0[best_dim] > threshold else '错误'}")
        print(f"  D3预测为{'D3' if fp1[best_dim] < threshold else 'D2'}  ← {'正确' if fp1[best_dim] < threshold else '错误'}")

        print(f"\n假设判定：{'可区分 D2/D3，路线2值得继续' if cos < 0.95 else '相似度过高，路线2可行性低'}")


if __name__ == "__main__":
    main()
