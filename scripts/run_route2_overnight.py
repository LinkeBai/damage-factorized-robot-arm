"""路线2完整流水线：基于集成分歧指纹的拓扑识别。

流程：
  1. 5 seeds × 训练普通集成
  2. 收集分歧指纹（每个 domain 10条轨迹）
  3. 训练拓扑分类器（指纹 → D2/D3/intact）
  4. 端到端评估：K 条探测轨迹 → 拓扑识别 → 条件化 WM → 预测改善
  5. 对比基线：K=0（不识别），oracle（真实拓扑）
  6. 生成完整报告

运行约 6-8 小时（5 seeds × 训练 + 评估）。

Usage:
  python scripts/run_route2_overnight.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.robotarm.envs.mujoco_env import MujocoArmEnv
from src.robotarm.training.topology_ensemble import (
    train_topology_ensemble,
    evaluate_topology_ensemble,
    encode_damage_batch,
    conditioning_damages,
)
from src.robotarm.training.sim_protocol import load_g1_protocol, damage_from_name
from src.robotarm.training.target_split import load_target_split
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains

CONFIG_PATH = Path("config/experiment/g2_push_ensemble_v1.yaml")
SEEDS = [7, 17, 27, 37, 47]
PROBE_TRAJS = 10   # 每个 domain 收集多少条指纹轨迹
K_SHOTS = [0, 1, 2, 5]
STATE_DIM = 14
STATE_NAMES = [
    "q1","q2","q3","q4","q5",
    "dq1","dq2","dq3","dq4","dq5",
    "block_x","block_y","block_vx","block_vy"
]
RESULTS_DIR = Path("results/final")
REPORTS_DIR = Path("reports")
RUN_DIR = Path("runs/route2_overnight")


# ── 分歧指纹计算 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_fingerprint(ensemble, domain, trajectories, joint_ranges, device, horizon=10):
    """返回 per-state-dim 平均分歧 (14,)。"""
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
            var = stacked.var(dim=0, unbiased=False).mean(dim=0)  # (state_dim,)
            per_dim_vars.append(var.cpu().numpy())

    return np.stack(per_dim_vars).mean(axis=0)  # (state_dim,)


# ── 拓扑分类器 ────────────────────────────────────────────────────────────────

class TopologyClassifier(nn.Module):
    """分歧指纹 (14,) → 拓扑类别（0=intact, 1=D2, 2=D3）。"""
    def __init__(self, input_dim=STATE_DIM, hidden_dim=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.label_map = {"intact": 0, "D2": 1, "D3": 2}
        self.inv_label_map = {0: "intact", 1: "D2", 2: "D3"}

    def forward(self, x):
        return self.net(x)

    def predict(self, fingerprint: np.ndarray, device) -> str:
        x = torch.tensor(fingerprint, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self(x)
        label_idx = int(logits.argmax(dim=-1).item())
        return self.inv_label_map[label_idx]


def train_classifier(fingerprints_by_domain, device, epochs=200):
    """用收集到的指纹训练分类器。"""
    X, y = [], []
    for domain_id, fps in fingerprints_by_domain.items():
        topo = domain_id.split("__")[0]
        label = {"intact": 0, "D2": 1, "D3": 2}.get(topo, -1)
        if label < 0:
            continue
        for fp in fps:
            X.append(fp)
            y.append(label)

    X_t = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # 标准化
    mean = X_t.mean(dim=0, keepdim=True)
    std = X_t.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_t = (X_t - mean) / std

    clf = TopologyClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        logits = clf(X_t)
        loss = F.cross_entropy(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 训练准确率
    with torch.no_grad():
        preds = clf(X_t).argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()

    return clf, mean, std, acc


# ── 端到端评估 ────────────────────────────────────────────────────────────────

def evaluate_with_topology_id(
    ensemble, clf, mean_norm, std_norm,
    domain, probe_trajs, eval_trajs,
    joint_ranges, device, k, horizon=10
):
    """K 条探测轨迹 → 指纹 → 拓扑识别 → 条件化 WM → RMSE。"""
    # Step 1: 从 K 条探测轨迹计算指纹
    if k == 0:
        predicted_topo = "intact"  # 无探测时默认 intact
    else:
        fp = compute_fingerprint(ensemble, domain, probe_trajs[:k], joint_ranges, device)
        fp_norm = (fp - mean_norm.cpu().numpy()) / std_norm.cpu().numpy().clip(1e-6)
        predicted_topo = clf.predict(fp_norm, device)

    # Step 2: 用识别出的拓扑条件化评估
    true_topo = domain.domain_id.split("__")[0]
    correct = (predicted_topo == true_topo)

    # 用预测拓扑的 damage 条件化
    predicted_damage = damage_from_name(predicted_topo)

    # 覆盖 domain 的 damage 用于评估
    states = torch.stack([t.states for t in eval_trajs]).to(device)
    actions = torch.stack([t.actions for t in eval_trajs]).to(device)
    damages = conditioning_damages([predicted_damage] * len(eval_trajs), "structured")
    contexts = [encode_damage_batch(m.encoder, damages, joint_ranges, device) for m in ensemble]

    horizon = min(horizon, actions.shape[1])
    sq_errs = []
    with torch.no_grad():
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
                mean_pred = torch.stack(means).mean(dim=0)
                target = states[:, start + offset + 1]
                sq_errs.append((mean_pred - target).pow(2).mean(dim=-1))

    rmse = float(torch.stack(sq_errs).mean().sqrt())
    return {
        "k": k, "ensemble_rmse": rmse,
        "predicted_topo": predicted_topo,
        "true_topo": true_topo,
        "correct_identification": correct,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_seed(seed, config, ranges, protocol, calibration_targets, evaluation_targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = int(config["steps"])
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. 训练结构化集成（condition_mode="structured"，使用拓扑 descriptor）
    print("[1/4] 训练结构化集成 …", flush=True)
    t0 = time.perf_counter()
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=seed * 10_000,
        targets=calibration_targets, excitation="goal", block_initial_xy=block_xy,
    )
    ensemble = train_topology_ensemble(
        train_trajs, ranges, members=3,
        epochs=int(config["epochs"]),
        device=device, seed=seed, condition_mode="structured",  # 用结构化集成
    )
    print(f"  完成 ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 2. 收集分歧指纹（训练集所有 domain）
    print("[2/4] 收集分歧指纹 …", flush=True)
    t0 = time.perf_counter()
    fingerprints_by_domain = {}
    all_domains = list(protocol.train) + list(protocol.test)
    for i, domain in enumerate(all_domains):
        probe_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=PROBE_TRAJS,
            steps=steps, seed=seed * 200_000 + i * 1000,
            targets=calibration_targets, excitation="active",
            block_initial_xy=block_xy,
        )
        fps = [compute_fingerprint(ensemble, domain, probe_trajs[j:j+1], ranges, device)
               for j in range(PROBE_TRAJS)]
        fingerprints_by_domain[domain.domain_id] = fps

    # 训练集 domain 的指纹
    train_fingerprints = {k: v for k, v in fingerprints_by_domain.items()
                          if any(k == d.domain_id for d in protocol.train)}
    print(f"  收集了 {len(train_fingerprints)} 个 domain 的指纹 ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 3. 训练拓扑分类器
    print("[3/4] 训练拓扑分类器 …", flush=True)
    clf, mean_norm, std_norm, train_acc = train_classifier(train_fingerprints, device)
    print(f"  训练准确率: {train_acc:.1%}", flush=True)

    # 验证测试 domain 上的分类准确率
    for domain in protocol.test:
        test_fps = fingerprints_by_domain.get(domain.domain_id, [])
        if not test_fps:
            continue
        correct = 0
        for fp in test_fps:
            fp_norm = (fp - mean_norm.cpu().numpy()) / std_norm.cpu().numpy().clip(1e-6)
            pred = clf.predict(fp_norm, device)
            if pred == domain.domain_id.split("__")[0]:
                correct += 1
        print(f"  {domain.domain_id}: 识别准确率 {correct}/{len(test_fps)} = {correct/len(test_fps):.1%}", flush=True)

    # 4. 端到端评估
    print("[4/4] 端到端评估 …", flush=True)
    rows = []
    for idx, domain in enumerate(protocol.test):
        probe_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=max(K_SHOTS),
            steps=steps, seed=seed * 100_000 + idx * 1000,
            targets=calibration_targets, excitation="active",
            block_initial_xy=block_xy,
        )
        eval_trajs = collect_push_domains(
            (domain,), trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps, seed=seed * 100_000 + idx * 1000 + 500,
            targets=evaluation_targets, excitation="goal",
            block_initial_xy=block_xy,
        )

        # 普通集成基线（K=0，intact 条件化）
        baseline = evaluate_topology_ensemble(
            ensemble, domain, eval_trajs, ranges, device=device,
            horizon=int(config["rollout_horizon"]), condition_mode="constant",
        )
        print(f"  {domain.domain_id} baseline(K=0): rmse={baseline['ensemble_rmse']:.4f}", flush=True)

        # 路线2：K条探测 → 拓扑识别 → 结构化条件化
        for k in K_SHOTS:
            if k == 0:
                r = {"k": 0, "ensemble_rmse": baseline["ensemble_rmse"],
                     "predicted_topo": "intact", "true_topo": domain.domain_id.split("__")[0],
                     "correct_identification": False}
            else:
                r = evaluate_with_topology_id(
                    ensemble, clf, mean_norm, std_norm,
                    domain, probe_trajs, eval_trajs, ranges, device,
                    k=k, horizon=int(config["rollout_horizon"]),
                )
            improvement = 100.0 * (baseline["ensemble_rmse"] - r["ensemble_rmse"]) / baseline["ensemble_rmse"]
            row = {
                "seed": seed,
                "domain": domain.domain_id,
                "method": "route2_topo_id",
                "k": k,
                "ensemble_rmse": r["ensemble_rmse"],
                "improvement_vs_baseline_pct": improvement,
                "predicted_topo": r["predicted_topo"],
                "true_topo": r["true_topo"],
                "correct_identification": r["correct_identification"],
            }
            rows.append(row)
            print(
                f"  K={k}: rmse={r['ensemble_rmse']:.4f}  "
                f"imp={improvement:+.2f}%  "
                f"pred={r['predicted_topo']}({'OK' if r['correct_identification'] else 'WRONG'})",
                flush=True
            )

    return rows


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration_targets = tuple(t.as_array() for t in targets.calibration)
    evaluation_targets = tuple(t.as_array() for t in targets.evaluation)
    ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("路线2：集成分歧指纹拓扑识别 — 完整 5-seed 运行", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)

    all_rows = []
    for seed in SEEDS:
        rows = run_seed(
            seed, config, ranges, protocol,
            calibration_targets, evaluation_targets,
        )
        all_rows.extend(rows)

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "route2_structured_topo_id_5seed.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n保存: {csv_path}", flush=True)

    # 汇总分析
    from collections import defaultdict
    by_domain_k = defaultdict(list)
    for row in all_rows:
        by_domain_k[(row["domain"], row["k"])].append(row)

    print("\n" + "="*60, flush=True)
    print("汇总结果", flush=True)
    print("="*60, flush=True)
    summary = {}
    for domain_id in ["D2__mixed_composition", "D3__mixed_composition"]:
        print(f"\n{domain_id}:", flush=True)
        for k in K_SHOTS:
            rows_dk = by_domain_k[(domain_id, k)]
            if not rows_dk:
                continue
            rmses = [r["ensemble_rmse"] for r in rows_dk]
            imps = [r["improvement_vs_baseline_pct"] for r in rows_dk]
            correct_ids = [r["correct_identification"] for r in rows_dk if k > 0]
            mean_rmse = np.mean(rmses)
            mean_imp = np.mean(imps)
            id_acc = np.mean(correct_ids) if correct_ids else float("nan")
            print(
                f"  K={k}: mean_rmse={mean_rmse:.4f}  "
                f"mean_imp={mean_imp:+.2f}%  "
                f"id_acc={id_acc:.1%}" if k > 0 else
                f"  K={k}: mean_rmse={mean_rmse:.4f}  (baseline)",
                flush=True
            )
            summary[f"{domain_id}_K{k}"] = {
                "mean_rmse": float(mean_rmse),
                "mean_improvement_pct": float(mean_imp),
                "identification_accuracy": float(id_acc) if k > 0 else None,
            }

    # 最终 gate 判定
    k5_imps = [
        summary.get(f"{d}_K5", {}).get("mean_improvement_pct", 0)
        for d in ["D2__mixed_composition", "D3__mixed_composition"]
    ]
    k5_id_accs = [
        summary.get(f"{d}_K5", {}).get("identification_accuracy", 0)
        for d in ["D2__mixed_composition", "D3__mixed_composition"]
    ]
    mean_k5_imp = np.mean(k5_imps)
    mean_k5_id_acc = np.mean(k5_id_accs)

    print(f"\nK=5 平均改善: {mean_k5_imp:+.2f}%", flush=True)
    print(f"K=5 平均识别准确率: {mean_k5_id_acc:.1%}", flush=True)

    if mean_k5_imp > 3.0 and mean_k5_id_acc > 0.6:
        gate = "GO"
        rationale = f"拓扑识别准确率 {mean_k5_id_acc:.1%}，K=5 改善 {mean_k5_imp:+.2f}%，方法论文可行"
    elif mean_k5_id_acc > 0.6:
        gate = "PARTIAL"
        rationale = f"识别准确率 {mean_k5_id_acc:.1%} 但预测改善不显著，需进一步分析"
    else:
        gate = "NO-GO"
        rationale = f"识别准确率 {mean_k5_id_acc:.1%} 不足，路线2不可行，转 benchmark 定位"

    print(f"\nGate: {gate} — {rationale}", flush=True)

    # 保存 JSON
    json_out = RESULTS_DIR / "route2_structured_topo_id_5seed.json"
    json_out.write_text(json.dumps({
        "experiment": "route2_topo_id",
        "seeds": SEEDS, "k_shots": K_SHOTS,
        "summary": summary,
        "mean_k5_improvement_pct": float(mean_k5_imp),
        "mean_k5_id_accuracy": float(mean_k5_id_acc),
        "gate": gate, "rationale": rationale,
    }, indent=2), encoding="utf-8")

    # Gate report
    date_str = date.today().strftime("%Y%m%d")
    report_path = REPORTS_DIR / f"route2-topo-id-gate-{date_str}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"""# 路线2 Gate 报告：集成分歧指纹拓扑识别

**日期**: {date_str}
**假设**: 集成成员的 per-dimension 分歧指纹携带故障拓扑信息，K 条主动探测轨迹可识别锁定关节

## 核心结果

| 指标 | 值 |
|---|---|
| K=5 平均改善 | {mean_k5_imp:+.2f}% |
| K=5 识别准确率 | {mean_k5_id_acc:.1%} |
| **Gate 决定** | **{gate}** |

## 决策理由

{rationale}

## 各 K 值结果

| Domain | K | RMSE | vs K=0 | 识别准确率 |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {d} | {k} | {summary.get(f'{d}_K{k}', {}).get('mean_rmse', 0):.4f} | "
        f"{summary.get(f'{d}_K{k}', {}).get('mean_improvement_pct', 0):+.2f}% | "
        f"{summary.get(f'{d}_K{k}', {}).get('identification_accuracy') or 'N/A'} |"
        for d in ["D2__mixed_composition", "D3__mixed_composition"]
        for k in K_SHOTS
    ), encoding="utf-8")
    print(f"报告保存: {report_path}", flush=True)


if __name__ == "__main__":
    main()
