# 本机实验进展记录

> 记录本机（RTX 3070 Laptop 8GB）实验进展与结论，供会话重建续接。只记录本机实际做过、已验证的内容。

## 摘要

**2026-08-18 正式复核**：六方法、五 seeds 的 Push 数值复现了 DFWM 相对 topology-only 的 15.8% 多步误差优势，但 95% CI 跨 0，且 DFWM 从 K=0 到 K=5 没有改善。因此该结果不能证明少样本 residual calibration 机制，当前 G1 判定为 No-Go/Pivot。详见 `reports/g1-push-formal-gate-20260818.md`。

**2026-08-18 主动校准修复**：加入训练期主动探针、强正则、latent 限幅和独立 validation 回退后，seed 7 的 K 曲线恢复单调改善，K=5 相对 K=0 改善 1.28%。该修复消除了破坏性过拟合，但低于 5% 小闸门，因此没有继续跑 3 seeds。详见 `reports/g1-active-calibration-diagnostic-20260818.md`。

**2026-08-18 目标导向 Push 复核**：改用 Push 专用 split、共同接触工作区、真实接触统计和“接近后推块”轨迹后，zero-shot DFWM 在 seed 7/17 均差于 topology-only（平均绝对差约 -0.0782）。由于 2/3 方向闸门已不可能通过，seed 27 按规则停止。原 15.8% 随机激励结果不能作为目标导向 Push 主结果。详见 `reports/g1-push-zero-shot-smoke-20260818.md`。

**历史一句话（已被正式复核修正）**：最初认为"因子化世界模型（DFWM）在机械臂关节锁定后能靠少量校准数据改善预测"，但经过反复验证发现——在简单的 Reach（末端到点）任务上，这个优势根本不存在（是训练过拟合的假象）；换到 Push 后曾观察到多步预测误差比纯拓扑基线好 15.8%。

**一句话后续**：在 Push 上做完整 6 方法对比 + 显著性检验，确认优势稳定后作为论文主结果；真机暂不可用，论文以仿真为主。

---

## 环境信息

| 项目 | 值 |
|---|---|
| GPU / Python / PyTorch / MuJoCo | RTX 3070 8GB / 3.10.20 (conda `mcp_env`) / 2.10.0+cu128 / 3.11.0 |
| 测试 | `pytest -q` → **114 passed** |

**本地偏离上游的改动（提交前需处理）**：
- `pyproject.toml` 的 `requires-python` 从 `>=3.11` 临时改为 `>=3.10`（复用 conda 里已装好的 torch）。
- Git 通过系统代理 `127.0.0.1:7890` 克隆（`git config --global http.proxy` 已配置）。

---

## 最终结论（当前最可信）

### 1. Reach 任务上 DFWM 无优势 —— 已彻底验证

在 5-DoF 自由空间 Reach 预测上，DFWM 相对零样本 `topology_only` 基线**没有实质优势**，无论：
- 用 one-step 还是 multi-step 指标；
- 加不加 backlash（舵机回差）等非线性 residual。

5 seeds 结果（K=5）：topology_only 反而最好，dfwm 略差（~-1%），差异在噪声范围内。

**根因（三层）**：① Reach 动力学太简单，锁定哪个关节（topology）已决定绝大部分 dynamics；② 8 维静态 z 无法捕获历史依赖的 residual；③ residual 模拟对 state 影响太小。

**被推翻的早期结论**：最初 3 seeds 显示"dfwm 显著优于 baseline（vs monolithic 好 18%）"是**假象**——是 baseline 也在过拟合、而 3 seeds 恰好都落在"训练稳定"的 seed 上造成的。

### 2. Push 任务上 DFWM 恢复优势 —— 当前主方向 ✅

换到 Push（推方块，接触动力学）后，residual 影响大一个量级，dfwm 的因子化建模真正体现价值：

**5 seeds Push 结果（K=5）：**

| 指标 | dfwm | topology_only | dfwm 优势 |
|---|---|---|---|
| one-step RMSE | 0.0349 | 0.0377 | +7.4% |
| **multi-step RMSE** | **0.1589** | **0.1888** | **+15.8%** |

**关键验证**：nominal 能推动方块 0.298 米，而 mixed_unseen（actuator 衰减 + 高阻尼 + 延迟 + backlash）**完全推不动**（位移 0）——residual 对 Push 的影响远大于 Reach。

---

## 实验历程（压缩时间线）

| 阶段 | 结果 | 状态 |
|---|---|---|
| V7 控制 hybrid 探索（IK+PD + WM 辅助） | WM 增益仅 ~1%，不显著 | 已放弃作为主结果 |
| 补 6 方法 baseline | 完成（topology/history/param-matched/residual/monolithic/dfwm） | 保留 |
| 3 seeds 预测基准 | dfwm"最优" | **假象，被推翻** |
| 方向 A（history encoder 改 amortized 变体） | 修复 K=0 公平性 | 保留 |
| held-out split（D4 + mixed_unseen） | "非记忆"检验 | 保留 |
| 5 seeds + 显著性检验 | dfwm 优势消失 | **关键转折** |
| early stopping + lr 调度 | 修复过拟合，揭穿无优势 | 保留 |
| 加 backlash | Reach 上仍无优势 | 保留（Push 里有用） |
| multi-step 评估 | Reach 上仍无优势 | 保留（Push 里关键） |
| **换 Push 任务** | **dfwm multi-step 优势 15.8%** | **当前主方向 ✅** |

---

## 当前代码/资产状态

**新增/修改的核心文件**：

| 文件 | 内容 |
|---|---|
| `sim/assets/arm_push.xml` | Push 环境（arm + 桌面 + 可推动方块） |
| `sim/assets/arm.xml` | Reach 环境（未动） |
| `src/robotarm/envs/mujoco_env.py` | 加 block 观测、backlash、state 14 维支持 |
| `src/robotarm/envs/residual_physics.py` | 加 `backlash` 字段 |
| `src/robotarm/models/history_encoder.py` | 新增（amortized residual 编码器） |
| `src/robotarm/training/g1_mechanism.py` | 6 方法训练、early stopping、lr 调度、multi-step 评估、state_dim 推断 |
| `src/robotarm/models/world_model.py` | state_dim 支持动态（14 维 Push） |
| `scripts/run_push_benchmark.py` | Push 预测基准（支持多 seed） |
| `scripts/analyze_seed_significance.py` | 逐 seed 分解 + bootstrap 显著性检验 |
| `scripts/merge_and_analyze_5seeds.py` | 合并 checkpoint + 显著性分析 |
| `config/splits/g1_5dof_heldout_v1.yaml` | 真正的 held-out split（D4 + mixed_unseen） |
| `paper/main.md` | 论文方法部分英文草稿 |

**6 方法 baseline**（计划书 §6.2 要求，已齐全）：
`topology_only`（零样本）、`history_encoder`（amortized）、`parameter_matched`（同参数不同结构）、`residual_only`（无拓扑）、`monolithic_matched`（单一 descriptor）、`dfwm`（主方法）。

---

## 遗留问题 / 待办

1. **Push 方向（最优先）**：补其他 baseline 的 multi-step 评估 + 逐 seed 显著性检验，确认 15.8% 优势稳定。
2. `pyproject.toml` 的 `requires-python` 本地改动，提交前处理。
3. 真机不可用 → 论文以仿真为主，真机作为未来工作。
4. `run_v7_k_shot_ablation.py` 有 bug（uncertainty 阈值 0.5 太高，rejection 100%），已不用于主实验。

## 后续计划（用户确认）

1. **Push 主结果做实**：6 方法完整 multi-step 对比 + 5 seeds 显著性检验 + bootstrap/CI。
2. 结果稳定后，作为论文新主结果，更新 `paper/main.md`（Push 实验部分）。
3. 真机 pilot（若恢复可用）或转投 RSS/CoRL。
