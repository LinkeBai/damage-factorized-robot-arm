# models/ 设计笔记 — DFWM 模型层实现蓝本

> 实现状态（2026-08-07）：M1-M3 与仿真版 M4/M5 smoke 已接通。
> 部署适应现在使用 `[e_topology, z_residual]` 条件化冻结世界模型；
> 真机 G0 校准仍是物理结论的硬前置。

> 目的：本文件是 `src/robotarm/models/*` 与 `src/robotarm/training/*` 的**待确认设计蓝本**。
> 严格对应 PROJECT-PLAN-V4 的 §2（假设）、§4（技术方案）、§6（实验/基线/指标）。
> 你确认后再按此写代码。标注 **[占位]** 的项依赖 G0 真机测量，当前给工程默认值。

---

## 0. 一句话架构

把"哪里坏了"（**离散**，诊断可得）与"坏得多离谱"（**连续 residual**，只能从少量真实交互中估计）**分开表示**：冻结 world model 与 policy，部署时仅从 1–5 条安全校准轨迹推断低维 `z_residual`，再闭环控制。

```
c_damage = [ e_topology(m, q_lock, joint_attributes) ,  z_residual(D_K) ]
              └──────── 离散 topology，固定 ────────┘   └─ 连续 residual，可推断 ─┘
```

---

## 1. 观测 / 状态约定（对接已有代码）

| 量 | 形状 | 来源 |
|----|------|------|
| `o_t` state | (12,) = qpos(6) + qvel(6) | `MujocoArmEnv._observe()` |
| `target` | (3,) | reset 传入 |
| `a_t` | (6,) 归一化 [-1,1] | policy 输出 |
| `m` 关节可用性 | (6,) 0/1 | `DamageConfig.joint_mask` |
| `q_lock` | (6,) | `DamageConfig.lock_angle` |
| `joint_attributes` | per-joint 稳定描述 | `arm.xml` / `arm_spec.yaml` **[占位]** |

观察规范：`state` + `target`（+ 可选 `image`），与 `Observation` 协议、schema 一致。
Reach 观测 **不含像素**（§ 对齐）；Push/Pick 再引入 object pose / image。

---

## 2. 四个核心构建块

### 2.1 Topology encoder（§4.2）→ `models/topology_encoder.py`

**输入**（每关节构造特征向量，禁止每关节独立 lookup embedding）：
```
per_joint_i = [ presence_i, lock_angle_i, axis_i(3), normalized_limits_i(2), depth_i ]
```
- `presence_i = m[i]`（是否锁定）
- `lock_angle_i = q_lock[i]`（非锁定关节置 0）
- `axis_i`：关节轴单位向量（从 arm.xml）
- `normalized_limits_i`：`[lo, hi]` 归一化到 [-1,1]
- `depth_i`：机械链层数
- 每关节维度 ≈ 1+1+3+2+1 = **8**

**处理**：共享 MLP 逐关节编码 → 按链序拼接（G1 就够，GNN/Transformer 非前置）→ 输出 `e_topology`（固定维，默认 64）。

> **约定**：`e_topology` **只由 m/q_lock/attributes 决定，不含任何数据**，因此预训练可覆盖大量 topology，**零样本即可用**（topology-only baseline 就是 z=0 直接用）。

---

### 2.2 Residual context 推断（§4.3）→ `models/residual_context.py`

三种实现按工程风险从低到高，代码里分目录/类：

| 变体 | 阶段 | 部署时推断 | 优点 | 缺点 |
|------|------|-----------|------|------|
| **A. Latent optimization** | **G1 默认** | 每个部署实例梯度优化 `z` | 简单、因果清晰 | 每次部署要梯度步 |
| **B. Amortized encoder** | G2 | 前馈，快 | 推断快 | 训练成本高 |
| C. Active calibration | G2+ 可选 | — | 主动信息增益 | 约束多，passive 无效才上 |

**A 的流程（与 §4.3 完全一致）**：
1. 初始化 `z_residual = 0 ∈ R^d`；
2. **冻结** WM 与 actor；
3. 最小化 K 条校准轨迹上的 **multi-step prediction loss**；
4. 只更新 `z_residual`。

> 关键取舍：latent opt 的部署成本 = 一次前向的梯度步数。计划将其视为"校准 wall-clock"指标，是方法主张的一部分（不只比准确率，也比部署成本）。

`d`（residual 维度）：候选 **{4, 8, 16}**，G1 默认 **8**，由 validation（非 test）选。

---

### 2.3 World model（§4.4）→ `models/world_model.py`

预测（recurrent，含 stochastic latent）：
```
p(o_{t+1}, r_t, continue_t | o_t, a_t, e_topology, z_residual)
```

**最低实现要求（照抄计划清单）**：
- deterministic recurrent state（GRU）✅
- stochastic latent（RSSM 风格采样）✅
- observation / reward / continue 三个 head ✅
- topology 与 residual **在 recurrent transition 与 prediction heads 都可访问** ✅
- 训练记录 one-step 与 multi-step error ✅
- 支持 actor-free 的 rollout prediction **smoke test** ✅（我们已能用 MujocoArmEnv 生成真实 rollout 对照）

> 这是 DFWM 的核心：`z_residual` 作为 transition 的条件输入，把"残差物理"编码进动力学，从而多步预测能吸收 backslash/compliance/latency。

---

### 2.4 Policy（§4.5）→ `models/actor_critic.py` + `models/planner.py`

两条路径：
1. **Dreamer-compatible actor-critic**：在 WM 的 latent imagination 上训练（主路径）
2. **MPC / short-horizon planner**：若 actor 训练不稳，用它验证 WM 本身（保险路径）

**G1 硬性要求（§4.5）**：至少有一个**冻结部署策略**。若只有 actor fine-tune 后能恢复 → 触发 **Pivot B**。这对我们的评估很重要：`evaluate.py` 目前打分的就是"冻结 policy 在这个 damage 下能不能靠推断 z 恢复"。

---

## 3. 训练分布（§4.6）— 依赖 G0，先留接口

计划明确：**不用未测量的单点物理常数**。以下参数 G0 后用分层区间确定，训练/验证/测试按组合切分：

- lock angle 范围
- locked-joint static/dynamic friction
- compliance / 等效弹簧
- backlash
- command latency
- payload
- servo tracking noise

**切分规则（写死进设计）**：
- 每个单参数水平在训练中出现；
- 某些 `joint × lock_angle × residual bin` 组合**只在测试出现**；
- test 不是简单同分布抽样；
- 切分写入**不可变 YAML + 记录哈希**（对应计划 §10.2 / storage 哲学）。

> 这一步现在**无法诚实地定死**，因为所有物理常数都依赖 G0。当前 MuJoCo 模型是**理想臂**（无 backlash/compliance），所以我们能跑通架构与 smoke，但**不能把当前仿真主表当成 ICRA 证据**——要等 G0 后把 MujoCo 参数域扩充到与真机一致。

---

## 4. Baselines（§6.2 必需）→ `models/baselines/`

| Baseline | 实现要点 | 公平性约束 |
|----------|---------|-----------|
| **Topology-only zero-shot** | 同一 WM、z=0 | 同 WM、同 topology |
| **History encoder** | RMA/OEA：短历史 (o,a,o') 序列 → 推断 | 同预训练数据、同观测历史、近似参数量 |
| **Matched continuous descriptor** | 单一连续 morphology descriptor | 同网络容量、同真实数据预算 |
| **Parameter-matched adaptation** | 增益只来自多可训练参数？ | 同 trainable params、同更新步数 |

代码约定：每个 baseline 暴露与 DFWM **相同的部署接口**（给定校准数据 → 得到可用的 policy），保证 `evaluate.py` 能统一打分。

---

## 5. 主指标（§6.5）→ `analysis/aggregate.py`

**首要指标 normalized recovery**：
```
NR = (S_adapted - S_no_adapt) / max(S_damaged_oracle - S_no_adapt, ε)
```
- `S` = 同一损坏条件下的 success 或 return
- **必须同时报告原始 success**，避免比例掩盖

其他指标（都已定义，`evaluate.py` 已覆盖 success rate / final dist / time-to-reach，其余待补）：
success rate、mean return、final position error、calibration transitions、calibration wall-clock、context optimization wall-clock、WM one/multi-step NLL、unsafe action / estop count、GPU-h、peak memory、trainable parameter count。

---

## 6. 统计方案（§6.6）

- **G1**：3 training seeds（方向/机制闸门，不做强显著性）
- **G2**：5 training seeds
- 仿真每 seed/condition **≥ 50 evaluation episodes**
- 消融（§6.4）：topology-only / residual-only / factorized；actor frozen vs head-updated；d=4/8/16；K=0/1/2/5/10；random vs held-out 组合切分；latent-opt vs amortized

---

## 7. 实现里程碑（建议顺序）

| 里程碑 | 内容 | 依赖 |
|--------|------|------|
| **M1** | `topology_encoder.py` + 单测 | 无（纯函数，可立即做） |
| **M2** | `residual_context.py`（variant A latent-opt）+ 单测 | M1 |
| **M3** | `world_model.py`（RSSM 最小）+ 单测 + rollout smoke | M1、M2、env |
| **M4** | `pretrain.py`：在仿真数据训 WM | M3、`collect_demo` 数据 |
| **M5** | 部署闭环：MujocoArmEnv + topology + latent-opt 推断 z + evaluate | M2–M4 |
| **M6** | baseline 集（topology-only / history / continuous descriptor） | M3、evaluate |
| **M7** | DFWM vs baselines 的 G1 机制表 + NR 指标 | M5、M6 |

**依赖说明**：M1–M3 **完全可离线**（不依赖真机）。M4 起需要真实物理参数才"有意义"，但**架构/smoke 先在理想仿真上跑通**（§4.6 强调：当前理想仿真 ≠ 正式证据）。

---

## 8. 待你确认的决策点

1. **WM 底层**：RSSM（Dreamer 风格，需 torch 实现 stochastic latent）？还是先用**确定性 recurrent + 高斯 head**（更简单，先把机制跑通，再升级 stochastic）？→ **建议：先 deterministic，M3 后再加 stochastic，符合"先跑通机制、G1 再加强"**
2. **d=8** 作为 G1 默认 residual 维度，validation 选 —— 认可吗？
3. **Reach 用纯 proprioception（无像素）**先跑 GF1 —— 认可吗？(计划主表 Reach 就是这个设定)
4. **M4 是否会因为"理想仿真不等于正式证据"而需要先并行等 G0** —— 你希望 M4 用什么数据？(a) 直接用当前理想仿真 先通管线；(b) 等 G0 参数域扩充后再训正式表 → **建议 (a) 通管线 + 留 G0 后重训**
5. **框架**：确认用 **PyTorch**（环境已装 torch 2.11+cu128）作为所有模型/训练的基石。

---

*蓝本基于 PROJECT-PLAN-V4 精读 + 仓库已有代码（protocol / MujocoArmEnv / schema / storage / evaluate / damage / reachability）。待你确认 §8 决策后开始实现。*
