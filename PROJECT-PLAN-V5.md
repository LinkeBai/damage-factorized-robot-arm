# Project Plan V5 — Damage-Factorized World Model

**项目**：六自由度低成本机械臂关节锁定后的少样本安全恢复
**版本日期**：2026-08-06
**规划模式**：standard  
**规划基线**：本文件是后续执行的最新基线；`PROJECT-PLAN-V3.md` 与 `EXECUTION-PLAN.md` 保留为历史记录  
**当前状态**：idea 已重新评审与重构；论文草稿存在；实验代码、仿真环境和真实结果尚不存在  
**证据约束**：本文中的时间、GPU-h、工程工时和阈值是项目管理估计，不是实验结果

---

## 0. 一页执行摘要

### 0.1 核心决策

旧路线“random mask + morphology token + actor-head fine-tuning”存在三个无法靠补实验解决的问题：

1. 训练覆盖测试 mask，不能支持“未见离散损坏适应”；
2. “连续 embedding 不能表示离散变化”不成立；
3. token 与 actor 同时更新，无法判断恢复来自哪个组件。

V4 将主线改为：

> **Damage-Factorized World Model（DFWM）**：将诊断可得的离散损坏拓扑，与必须从少量真实交互中估计的连续残余物理分开表示。部署时冻结 world model 和 policy，仅从 1–5 条安全校准轨迹推断低维 residual context，再进行闭环控制。

### 0.2 ICRA 2027 投稿约束与会议策略

| 目标 | 官方状态（2026-08-06 核验） | 项目策略 |
|---|---|---|
| **ICRA 2027** | 常规论文截止为 **2026-09-15 11:59 PM PST**；完整论文（正文、图表、表格、致谢、参考文献）总计最多 8 页；双栏、双匿名、PDF 投稿。配套视频最多 180 秒、20 MB，首个上传窗口为 2026-08-05 至 09-09，第二窗口为 09-17 至 09-22。[官方投稿说明](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) | **唯一主目标**；所有资源、证据和排期围绕该截止倒排 |
| **RSS / CoRL 2027** | 当前不参与本轮资源排期 | 仅作为 ICRA 未提交、被拒或主动撤退后的后续去向，不得稀释当前执行目标 |

ICRA 提交还必须满足：至少选择 3 个官方关键词；PaperPlaza 元数据与匿名稿一致；正文不得包含可识别作者身份的信息；除可选视频外，不假设审稿人会访问外部链接或补充材料；投稿前再次核验 PDF compliance 与 IEEE-RAS AI 使用披露要求。

### 0.3 总体路线

| 阶段 | 目标 | 主要成本 | 关键决策 |
|---|---|---:|---|
| G0 | 5-DoF+夹爪运动学、可达性、硬件与物理测量 | 16–24 工时；4–8 真机小时 | URDF 与真机是否一致、任务是否物理可做 |
| G1 | 最小机制验证 | 40–60 工时；30–60 GPU-h | factorization 是否值得继续 |
| G2 | ICRA 核心仿真证据 | 60–90 工时；60–100 GPU-h | 是否形成稳定方法贡献 |
| G3 | 真机重复验证 | 30–45 工时；8–16 真机小时；8–16 GPU-h | 是否支持真实恢复主张 |
| G4 | 论文、视频与投稿 | 30–45 工时 | 是否达到投稿完整性 |

**ICRA 冲刺预算**：约 176–264 工时、98–186 GPU-h、12–28 真机小时、75–240 GB 存储。
**成本控制原则**：G1 不通过，不进入 G2；G2 不通过，不投入正式真机统计。

---

## 当前执行状态（2026-08-10 更新）

### G0

G0 已完成并通过，交付物、真机校准、MuJoCo 模型、可达域、锁定安全、急停和 10 姿态 TCP 记录均已归档。G0 仍保留“后 5 个姿态为用户确认一致而非独立尺量”的证据说明。

### G1

**2026-08-19 更新**：在 Push 专用 split、D2/D3 共同接触工作区和目标导向轨迹下，完成 five-shot DFWM、受约束主动校准、zero-shot、residual descriptor 监督、history encoder 监督及 residual-only 3-seed Pivot。所有分支均未获得稳定的 2/3-seed 机制信号；当前 G1 判 No-Go，暂停 G2。随机激励场景中的 15.8% 平均差异保留为诊断结果，不作为 few-shot recovery 或目标导向 Push 结论。完整审计见 `reports/g1-overnight-method-audit-20260819.md`。

原始 G1 learned-MPC Go 未通过：3 seeds 的 D2/D3 控制成功极不稳定，只有 1 次成功。诊断确认 IK+PD 在相同 D2/D3 与 evaluation targets 上可成功，因此失败原因是 learned world model 长时滚动预测与直接 torque-CEM 控制不稳定，不是目标不可达、MuJoCo 或 GPU 问题。

按本 V5 框架完成了 G1-Pivot：

- IK+PD hybrid：D2/D3、8 targets，8/8 成功；
- Jacobian residual feedback：8/8 成功；
- gated world-model hybrid：3 seeds、D2/D3、K=0/5，共 48/48 成功；
- residual-aware option selector：3 seeds、D2/D3、24 episodes，24/24 成功。

因此当前项目状态为：**G1 原始方法 No-Go；G1-Pivot 功能验证完成；不阻塞后续 V6/G2 优化。**

需要保留的科学限制：option selector 平均约 83.4 步，IK+PD 基线约 72.9 步，当前没有证明显著性能增益。world model 已参与候选动作选择，但尚未证明它比简单 IK+PD 更有效。

## 1. 项目目标与成功定义

### 1.1 科学目标

研究低成本串联机械臂发生单关节锁定后，如何利用诊断信息与极少量安全交互，恢复对新目标的控制能力。

### 1.2 工程目标

交付一个可复现系统，包含：

- 5-DoF 机械臂加独立夹爪 MuJoCo 模型；
- URDF—舵机—真机坐标映射和经过实测校准的运动链；
- 可配置的关节锁定、摩擦、顺应性、背隙和延迟模型；
- 仿真与真机统一轨迹接口；
- factorized conditional world model；
- residual context 推断；
- 至少 Reach 和 Push 两个任务；
- 可重跑的实验配置、日志、checkpoint 和统计脚本；
- 真机校准协议、视频与安全记录。

### 1.3 项目级成功条件

项目达到“机器人顶会可投稿”必须同时满足：

1. **机制成立**：冻结 actor 和 WM 时，factorized context 相对 topology-only 仍有稳定收益；
2. **非记忆**：在训练未出现的 topology–residual 组合上仍有效；
3. **基线可信**：至少覆盖 topology-only、history encoder、matched continuous descriptor、parameter-matched adaptation；
4. **真实重复**：真机不是单段展示；最低 ICRA 证据包覆盖两个故障条件且每条件不少于 20 个 evaluation episodes，强证据包每条件不少于 30 个；
5. **成本透明**：报告交互步数、真实秒数、适配时间、GPU-h、失败次数；
6. **可复现**：从环境创建到表格生成存在单命令或明确脚本链；
7. **主张克制**：所有结论与结果严格对应，不使用未验证成功率或“first”式主张。

---

## 2. 研究问题、假设与非主张

### 2.1 问题设定

给定一个已训练的机械臂控制系统。部署时发生单关节锁定：

- 故障诊断模块或人工检查能够提供锁定关节身份；
- 锁定角可以直接读取或粗略测量；
- 摩擦、顺应性、回差、负载和延迟等 residual dynamics 未知；
- 只允许 1–5 条低风险校准轨迹；
- 评估目标与校准目标分离。

### 2.2 主要研究问题

- **RQ1**：离散 topology 与连续 residual 的因子化，是否比单一 morphology descriptor 更利于组合泛化？
- **RQ2**：在冻结 world model 和 actor 时，少量校准轨迹是否足以改善控制？
- **RQ3**：收益是否来自 residual identification，而非 actor-head 行为克隆？
- **RQ4**：仿真中得到的 residual inference 能否吸收低成本真机的背隙、锁定顺应性与延迟？
- **RQ5**：达到给定恢复水平需要多少轨迹、多少 transition 和多少 wall-clock time？

### 2.3 可证伪假设

- **H1 Factorization**：在 held-out topology–physics 组合上，DFWM 的 normalized recovery 高于 topology-only 和 monolithic descriptor。
- **H2 Low-shot calibration**：1–5 条轨迹内，DFWM 的性能随数据量稳定改善，并早于 full fine-tune 达到平台期。
- **H3 Attribution**：冻结 actor 后仍保留主要收益；若收益消失，则 token/context 机制主张失败。
- **H4 Sim-to-real residual**：真实校准后，world-model prediction error 下降，并与控制成功率改善相关。
- **H5 Safety/cost**：DFWM 不需要比 history baseline 更多的真实交互或更多不安全动作。

### 2.4 明确不再主张

- 不主张连续向量无法表达离散故障；
- 不主张随机 mask 本身是新算法；
- 不把训练中出现过的 joint mask 称为 unseen morphology；
- 不把 actor-head fine-tuning 的收益归因给 morphology token；
- 不把 intact robot 表现当作 damaged morphology 的唯一 oracle；
- 不在真实数据产生前承诺 60%、80% 或固定胜幅；
- 不把“低成本平台”本身当作算法新颖性。

---

## 3. 研究范围与降维策略

### 3.1 主线范围

| 维度 | 主线 |
|---|---|
| 平台 | 现有 5-DoF GenkiArm 3D 打印臂 + 独立夹爪 + Feetech STS3215 |
| 故障 | 单关节锁定 D2 / D3 / D4 |
| 任务 | Reach、Push |
| 观测 | proprioception + 任务状态；RGB 仅用于目标/物体定位与视频 |
| 动作 | 5 维关节位置/增量命令；锁定关节动作由 adapter 屏蔽；夹爪开合作为独立执行器，不计入 5-DoF 定位链 |
| 仿真 | MuJoCo |
| 真机 | Feetech SDK + Python + websockets；不要求 ROS |
| 模型 | 小型 conditional RSSM / DreamerV3-compatible WM |
| 部署更新 | residual context；WM 与 actor 默认冻结 |

### 3.1.1 真实机械链与命名冻结

URDF `genkiarm.urdf` 给出的串联链为：

```text
Base --J1--> Yao --J2--> Jian1 --J3--> Jian2
     --J4--> Wan --J5--> Wan1 --J6--> Zhua
```

| 统一编号 | URDF joint | 功能角色 | URDF 轴 | 名义 origin xyz (m) | 主实验状态 |
|---|---|---|---|---|---|
| J1 | `Rotation` | 底座旋转 | X（joint frame） | `[-0.013, 0, 0.0265]` | 完整建模；不作为首批锁定故障 |
| J2 | `Rotation1` | 中间关节 1 | Y | `[0.081, 0, 0]` | D2 主故障 |
| J3 | `Rotation2` | 中间关节 2 | Y | `[0, 0, 0.118]` | D3 主故障 |
| J4 | `Rotation3` | 中间关节 3 | Y | `[0, 0, 0.118]` | D4 主故障 |
| J5 | `Rotation4` | 手腕 | Z | `[0, 0, 0.0635]` | 完整建模；姿态故障扩展 |
| J6 | `Rotation5` | 夹爪整体姿态 | X | `[0, -0.0132, 0.021]` | 完整建模；姿态故障扩展 |

“夹爪自由度”在本文中指 J6 对夹爪整体姿态的控制。若实体夹爪还有手指开合电机，则记录为 `gripper_open` 独立执行器，不将其误计为第七个机械臂定位自由度。URDF 中所有 ±1.57 rad 仅为名义限位，真实软限位必须经 G0 测量后覆盖。

### 3.2 延后项目

以下内容只有在 G2 通过后才允许加入：

- Pick/Place；
- RGB end-to-end world model；
- 双关节损坏；
- 未知故障身份的在线诊断；
- uncertainty-aware active probing；
- 多机械臂或多 embodiment；
- GNN morphology encoder；
- LoRA 大矩阵；
- ROS/MoveIt 集成。

### 3.3 删除或替换

- `From-scratch-5 RL` 替换为 `BC-from-5` 或 `offline learner-from-5`；
- “Danesh exact reproduction”替换为 matched continuous-descriptor baseline；
- 8-baseline × 9-cell 全量矩阵缩减为 4 个因果 baseline × 6 个主 cell；
- Pick 仅在锁定后可达性达到门槛时进入附加实验。

---

## 4. 技术方案

### 4.1 损坏上下文

定义：

```text
c_damage = [e_topology(m, q_lock, joint_attributes), z_residual(D_K)]
```

其中：

- `m ∈ {0,1}^6`：六个定位关节的可用性；
- `q_lock`：锁定关节角度；
- `joint_attributes`：关节轴、真实范围、父子拓扑、链深度及 `base/intermediate/wrist/gripper-orientation` 功能角色；
- `D_K`：K 条校准轨迹；
- `z_residual ∈ R^d`：未知物理残差的低维表示。

`d` 的初始候选为 4、8、16；G1 默认使用 8，最终由 validation 而非 test 选择。

### 4.2 Topology encoder

禁止只使用“每个关节一个独立 lookup embedding”作为最终方案，因为它难以支持组合解释。推荐：

1. 每关节构建 `[presence, lock_angle, axis, normalized_limits, depth]`；
2. 共享 MLP 编码每个关节；
3. 按机械链顺序拼接，或用轻量 attention/pooling；
4. 输出固定维度 `e_topology`。

G1 可先实现按顺序拼接的共享 MLP；GNN/Transformer 不是前置条件。

### 4.3 Residual context 推断

按工程风险从低到高实施：

#### A. Latent optimization（G1 默认）

- 为每个部署实例初始化 `z_residual = 0`；
- 冻结 WM 和 actor；
- 最小化校准轨迹上的 multi-step prediction loss；
- 只更新 `z_residual`；
- 优点：实现简单，因果归因清楚；
- 缺点：每次部署需要梯度步骤。

#### B. Amortized encoder（G2）

- 输入最近的 `(o_t, a_t, o_{t+1})` 序列；
- 输出 residual posterior 的均值和方差；
- 可用 latent-optimization 结果作为训练 target 或直接端到端训练；
- 优点：推断快；
- 缺点：实现和训练成本更高。

#### C. Active calibration（可选 G2+）

- 从安全动作库中选最大化预测分歧或 posterior information gain 的动作；
- 必须满足软限位、速度、温度/电流和工作区约束；
- passive calibration 无效时不得直接上此模块。

### 4.4 World model

世界模型预测：

```text
p(o_{t+1}, r_t, continue_t | o_t, a_t, e_topology, z_residual)
```

最低实现要求：

- deterministic recurrent state；
- stochastic latent；
- observation、reward、continue heads；
- topology/residual 在 recurrent transition 与 prediction heads 均可访问；
- 训练期间记录 one-step 与 multi-step error；
- 支持 actor-free 的 rollout prediction smoke test。

### 4.5 Policy

首选两种实现路径：

1. **Dreamer-compatible actor-critic**：与 world model latent imagination 一致；
2. **MPC/short-horizon planner**：若 actor 训练不稳定，可用于验证 world model 本身。

G1 必须至少有一个冻结部署策略。若只有 actor fine-tuning 后能恢复，则项目自动触发 Pivot B。

### 4.6 训练分布

训练 domain 不使用未经测量的单点物理常数。G0 后根据真机测量确定：

- lock angle 范围；
- locked-joint static/dynamic friction；
- compliance 或等效弹簧参数；
- backlash；
- command latency；
- payload；
- servo tracking noise。

每个参数使用分层区间。训练/验证/测试按组合切分：

- 每个单独参数水平在训练中出现；
- 某些 `joint × lock angle × residual bin` 组合只在测试出现；
- test 不是简单从同一分布随机抽样；
- 切分写入不可变 YAML 并记录哈希。

### 4.7 真机与仿真 adapter

统一接口：

```python
class RobotEnv(Protocol):
    def reset(self, *, target, damage_config) -> Observation: ...
    def step(self, action) -> StepResult: ...
    def emergency_stop(self) -> None: ...
    def close(self) -> None: ...
```

仿真实现 `MujocoArmEnv`；真机实现 `FeetechArmEnv`。训练代码不得直接依赖 Feetech SDK 或 MuJoCo API。

---

## 5. 任务与故障协议

### 5.1 Reach

- 目标：末端到达 3D 目标位置；
- 主指标：成功率、最终距离、到达时间；
- 初始成功阈值沿用 5 cm 仅作工程起点，G0 根据相机和运动学误差校准；
- calibration targets 与 evaluation targets 不重合；
- target 只从健康与损坏 morphology 的共同可达域采样；
- 主指标采用 position-only Reach，末端姿态作为次指标，避免把 J5/J6 的姿态能力与 J2–J4 的位置可达能力混为一谈。

### 5.2 Push

- 目标：推动方块进入目标区域；
- 主指标：成功率、最终物体距离、碰撞/越界；
- 必须先固定物体尺寸、摩擦面和相机标定；
- 评估分 easy/medium/hard target bins，但主结论预先选定一个 aggregate。
- Push 除位置误差外记录夹爪/末端接触姿态；若 J5/J6 未被控制稳定，不进入正式主表。

### 5.3 Pick（条件性）

进入条件：

- 锁定后 position-only 可达率足够；
- 抓取器与物体检测已稳定；
- 健康策略在仿真和真机均达到可重复基线；
- 不影响主线日期。

未通过则删除，不将不可达任务失败解释为适应失败。

### 5.4 故障条件

- D1：J1 底座锁定，主要压缩方位工作区，仅作扩展；
- D2：J2 中间关节 1 锁定，主故障；
- D3：J3 中间关节 2 锁定，主故障；
- D4：J4 中间关节 3 锁定，主故障；
- D5：J5 手腕锁定，主要影响末端姿态，仅作扩展；
- D6：J6 夹爪整体姿态锁定，仅作扩展；
- 每种故障至少包含多个 lock angles；
- “软件把动作设为 0”与“高摩擦物理锁定”必须分开；
- 真机 screw fixation 的实际微动由测量决定，不沿用 V3 的假设值。

ICRA 主实验固定为 D2/D3/D4，因为它们直接改变位置可达性、冗余与动力学，且三者具有相同的中间关节功能族，便于公平比较。D1 与 D5/D6 的故障后任务定义不同，不与主结果平均；只有主证据包完成后才作为边界分析。

---

## 6. 实验与证据矩阵

### 6.1 核心方法

- DFWM + passive residual calibration；
- 可选 DFWM + amortized residual encoder；
- 可选 DFWM + active calibration。

### 6.2 必需 baseline

| Baseline | 回答的问题 | 公平性 |
|---|---|---|
| Topology-only zero-shot | 真实轨迹是否必要 | 同一 WM、同一 topology、`z=0` |
| History encoder | RMA/OEA 类短历史推断是否已足够 | 同预训练数据、同观测历史、近似参数量 |
| Matched continuous descriptor | factorization 是否比单一连续 descriptor 有效 | 同网络容量、同真实数据预算 |
| Parameter-matched adaptation | 收益是否只是多了可训练参数 | 同 trainable params、同更新步数 |

### 6.3 条件 baseline

- Full fine-tune：仅当 G1 表明小 context 有优势时加入；
- BC-from-5：用于识别是否只是 few-shot imitation；
- Damaged oracle：获得真实 residual 参数或充分损坏数据；
- Intact oracle：只作为健康参考，不作为损坏上限；
- LoRA：仅当 reviewer 风险仍指向 generic adapter 时加入。

### 6.4 必需消融

1. topology-only；
2. residual-only；
3. factorized；
4. actor frozen vs actor-head updated；
5. latent dimension 4/8/16；
6. K = 0/1/2/5/10；
7. random combination split vs held-out combination split；
8. latent optimization vs amortized encoder（G2）。

### 6.5 主要指标

定义 normalized recovery：

```text
NR = (S_adapted - S_no_adapt) / max(S_damaged_oracle - S_no_adapt, ε)
```

其中 `S` 为同一损坏条件下的 success 或 return。必须同时报告原始 success，避免比例掩盖。

其他指标：

- success rate；
- mean return；
- final position/object error；
- calibration transitions；
- calibration wall-clock；
- context optimization wall-clock；
- WM one-step / multi-step NLL；
- unsafe action / emergency-stop count；
- GPU-h；
- peak memory；
- trainable parameter count。

### 6.6 统计方案

- G1：3 training seeds，用于方向与机制闸门，不做强显著性主张；
- G2：5 training seeds；
- 仿真每个 seed/condition 至少 50 evaluation episodes；
- 真机最低包每 condition 至少 20 episodes；强包至少 30 episodes，并跨至少 3 个 target sets 或 3 个实验日；
- 使用 hierarchical paired bootstrap：先重采样 seed，再重采样 seed 内 targets；
- 主要比较预先限定为 3 个，必要时使用 Holm correction；
- 报告效应量与 95% CI；
- 不能把同一训练模型的多次 rollout 当作独立训练重复；
- 不用 5 seeds 的双侧 Wilcoxon 星号支撑核心结论。

---

## 7. 阶段门与 Definition of Done

## G0 — 物理与可达性基线

**建议日期**：2026-08-06 至 2026-08-11
**负责人类型**：项目本人；必要时机械/控制同学复核  
**后续 owner**：`ccf-experiment-designer` 负责将测量结果固化为最终实验协议

### 输入

- 实体机械臂；
- 现有网页控制接口；
- 相机；
- STS3215 规格与 SDK；
- `genkiarm.urdf`；
- URDF 引用的 7 个 STL 网格或可测量的等效碰撞几何；
- 3D 模型或可测量连杆尺寸。

### 任务

1. 建立 J1–J5 与舵机 ID、URDF joint、控制命令通道及夹爪 ID6 的唯一映射；
2. 记录 6 个关节的真机零位、方向、软限位、最大安全速度和命令单位；
3. 修正 URDF 的 XML 兼容性，补齐 mesh 路径；禁止把缺少 inertial/collision/dynamics 的原始 URDF 直接当作动力学真值；
4. 测量连杆长度、末端偏置、夹爪 TCP；确认 J6 是夹爪整体姿态轴，并单独记录夹爪开合执行器；
5. 建立 FK/数值 IK，以至少 10 个非奇异姿态对照真机 TCP 测量；
6. 分别计算 intact、D2、D3、D4 的 position-only 共同可达域，以及含姿态约束的共同可达域；
7. 测量每个关节自由状态及 D2/D3/D4 锁定状态下的：
   - 稳态位置误差；
   - step response；
   - 回差；
   - 锁定角微动；
   - 命令—响应延迟；
   - 电流/温度可读性；
8. 确定 emergency stop；
9. 按底座、中间关节、腕部/夹爪姿态分别形成安全动作边界；
10. 决定 Reach/Push/Pick 的保留范围。

### 交付物

- `hardware/arm_spec.yaml`；
- `hardware/joint_map.yaml`（J1–J5、夹爪、URDF、舵机 ID、方向、零位、单位）；
- `hardware/calibration/` 原始数据；
- `hardware/safety_limits.yaml`；
- `sim/assets/genkiarm_calibrated.urdf` 与 `sim/assets/arm.xml` 初版；
- `reports/urdf-gap-report.md`（mesh、inertial、collision、dynamics 缺口）；
- `reports/g0-feasibility.md`；
- reachability 图与 target split。

### Pass

- 六关节命令映射无歧义，FK 端点误差满足任务容差需求；
- D2/D3 至少两个故障存在足够共同可达域；
- 锁定方式可重复且不导致危险电流/温升；
- 能在 10–20 分钟内重复安装和解除锁定；
- emergency stop 已测试。

### Block / Stop

- 锁定方式损坏舵机或不可重复；
- D2/D3 共同可达目标过少；
- 无法获得稳定关节状态；
- 无可靠急停。

## G1 — 最小机制验证

**建议日期**：2026-08-10 至 2026-08-23
**依赖**：G0 的尺寸、范围和最低物理参数  
**预算**：30–60 GPU-h；40–60 工时

### 固定范围

- Reach；
- D2、D3；
- 3 seeds；
- 4 方法：topology-only、residual-only/history、factorized、actor-head/parameter-matched；
- state observation；
- passive calibration；
- K = 0/1/2/5。

### 交付物

- 可运行 MuJoCo 环境；
- 100-step smoke test；
- dataset generator；
- 最小 conditional RSSM；
- residual latent optimization；
- 冻结 actor 或 MPC；
- G1 结果表、恢复曲线、prediction error；
- 每 run 的 manifest 与日志。

### Go

以下是项目闸门，不是论文结论：

- D2、D3 中 factorized 相对 topology-only 的 NR 改善方向一致；
- 至少 2/3 seeds 改善；
- actor 冻结时收益仍存在；
- K 增加时 prediction error 与控制表现总体改善；
- 没有不可解释的 data leakage。

### Pivot

- factorized ≈ topology-only：转为 robust zero-shot / benchmark；
- actor-head update 才有效：转为 few-shot imitation；
- history encoder 明显更好：将 DFWM 降为 baseline，转向在线 diagnosis；
- WM 不稳定但 MPC dynamics 有效：改为 conditional dynamics + MPC，放弃 Dreamer 品牌。

## G2 — 主会级仿真

**建议日期**：2026-08-24 至 2026-09-04
**依赖**：G1 Go  
**预算**：60–100 GPU-h；60–90 工时

### 固定范围

- Reach + Push；
- D2/D3/D4；
- 最低 ICRA 包 3 seeds；强证据包在算力允许时扩展至 5 seeds；
- 4 必需 baseline；
- held-out composition；
- K = 0/1/5（主文）；K = 2/10 仅在不影响关键路径时补充；
- factorization、actor-freeze 为必需消融；latent dimension 与 inference-method 大扫描降为非关键扩展。

### 交付物

- 主结果表；
- calibration curve；
- held-out composition 表；
- mechanism ablation；
- prediction/control correlation；
- robustness：backlash、delay、payload；
- failure taxonomy；
- compute table。

### Pass

- 主效应不是单一 seed 或单一 damage 驱动；
- held-out 组合上保持实质收益；
- 参数量与数据预算公平；
- 至少一个 negative/failure regime 被清晰识别；
- 完整运行成本不超过重新批准的预算。

### Stop

- factorization 在 G2 扩展后效应消失；
- damaged oracle 与 no-adapt 接近，说明任务本身不可恢复；
- baseline 无法公平实现；
- 仿真参数对结论极端敏感且不能由真机测量约束。

## G3 — 真机重复验证

**建议日期**：2026-08-24 至 2026-09-06，与 G2 并行
**依赖**：G1 Go；D3 真机 pilot 不等待完整 G2，正式统计依赖核心仿真趋势稳定
**预算**：8–16 真机小时；8–16 GPU-h；30–45 工时

### 顺序

1. intact Reach；
2. D3 topology-only；
3. D3 factorized calibration；
4. 第二 lock angle 或 D2；
5. Push；
6. 只有前述稳定后才做扩展视频。

### 交付物

- 原始轨迹；
- calibration/evaluation 明确分离；
- 每次实验安全日志；
- 最低包每 condition ≥20 episodes；强证据包每 condition ≥30 episodes；
- 至少两个故障条件；
- 成功/失败视频；
- sim-to-real error report。

### Pass

- 至少两个 condition 中收益方向一致；
- 结果跨天或跨 target set 可重复；
- 未发生不可接受的安全事件；
- 真实 residual inference 确实降低 prediction error；
- 视频与数值结果一致。

## G4 — 论文与投稿

**建议日期**：2026-08-17 至 2026-09-15，与 G1–G3 并行
**依赖**：方法与实验设置可提前写；结果主张依赖冻结的 G2/G3 证据

### 交付物

- 重写后的 `paper/main.md` 与 `paper/main.tex`；
- 图表 source data；
- 补充视频；
- 代码 README；
- 环境与 checkpoint；
- integrity audit；
- submission check；
- 公开前隐私、安全和许可检查。
- 8 页完整稿页数预算、匿名检查、至少 3 个 ICRA 关键词与 PaperPlaza 元数据；
- 不超过 180 秒、20 MB 的匿名配套视频及上传回执。

### Pass

- 摘要、贡献和结论无预期结果；
- 所有数字可追溯到不可变结果文件；
- 表格与正文一致；
- closest work 更新到投稿前；
- 会议格式、匿名和页数符合官方规则；
- 代码/数据公开范围明确。

---

## 8. ICRA 2027 主目标倒排与硬闸门

ICRA 2027 是本轮唯一主目标。闸门用于控制主张和实验范围，而不是把会议重新降级为备选：

| 日期 | 必须完成 |
|---|---|
| 2026-08-08 | J1–J5/夹爪舵机映射冻结；原始 URDF 缺口清单完成；急停验证 |
| 2026-08-11 | G0 通过；校准后的 5-DoF MuJoCo arm 可加载；intact/D2/D3/D4 可达域完成 |
| 2026-08-16 | Reach intact/D3 baseline 稳定；数据管线可重跑；论文问题定义和方法初稿更新 |
| 2026-08-23 | G1 通过；factorized 在冻结 actor 时有方向一致收益；至少一个 D3 真机闭环 pilot |
| 2026-08-30 | D2/D3/D4 核心仿真表与公平基线完成；第二个真机 condition 开始统计；视频素材可用 |
| 2026-09-04 | 主文数值冻结；最低真机证据包完成；8 页匿名稿完整 |
| 2026-09-08 | 最终视频完成并在首个窗口关闭前上传；主图表与 failure cases 冻结 |
| 2026-09-10 | 内部审稿、引用/数字/匿名/格式核验完成 |
| 2026-09-12 | PaperPlaza 元数据、关键词、PDF compliance 预检查；预留三天缓冲 |
| 2026-09-15 | 11:59 PM PST 前正式提交 PDF |

若闸门未通过，首先降级非核心范围，而不是伪造完成状态：Push 可降为附加结果；5 seeds 可降至 3 seeds 并报告不确定性；K=2/10、latent-dimension 全扫描和 amortized encoder 可删除。以下行为始终禁止：

- 填预期数字；
- 把单次 demo 当统计；
- 删除失败结果；
- 弱化 baseline；
- 将 G1 小规模结果包装成完成的 G2；
- 用仿真结果替代真机主张，或把校准轨迹重复计入 evaluation。

### 8.1 Minimum Viable ICRA Submission

- 5-DoF+夹爪校准运动学与安全协议完整；
- Reach 主任务，D2/D3/D4 仿真，至少 3 seeds；
- topology-only、history encoder、matched descriptor、parameter-matched 四个公平基线；
- factorization 与 actor-freeze 两个关键消融；
- 至少两个真机故障条件，每条件不少于 20 个独立 evaluation episodes；
- 1–5 条校准轨迹与 evaluation targets 严格分离；
- 8 页匿名论文、可追溯表图和合规视频。

### 8.2 Strong ICRA Submission

在最低包之上增加：Push、5 seeds、每真机 condition ≥30 episodes、held-out composition 完整矩阵、prediction/control correlation、跨实验日重复和明确 failure regime。强包不得阻塞最低包按时冻结。

### 8.3 八页主文预算

| 内容 | 目标页数 |
|---|---:|
| 摘要 + 引言 + 贡献 | 1.0 |
| 相关工作 | 0.6 |
| 问题定义与 5-DoF 故障设置 | 0.7 |
| DFWM 方法 | 1.5 |
| 实验协议与真实平台 | 1.0 |
| 主结果、消融、真机结果 | 2.2 |
| 局限、结论、致谢 | 0.4 |
| 参考文献 | 0.6 |

总计目标 8.0 页；参考文献也计入上限。最终分页以官方模板编译结果为准。

---

## 9. 推荐代码结构

```text
robotarm/
├── pyproject.toml
├── README.md
├── PROJECT-PLAN-V4.md
├── config/
│   ├── base.yaml
│   ├── env/
│   ├── model/
│   ├── experiment/
│   └── splits/
├── src/robotarm/
│   ├── envs/
│   │   ├── protocol.py
│   │   ├── mujoco_env.py
│   │   ├── feetech_env.py
│   │   ├── tasks.py
│   │   ├── damage.py
│   │   └── safety.py
│   ├── models/
│   │   ├── topology_encoder.py
│   │   ├── residual_context.py
│   │   ├── world_model.py
│   │   ├── actor_critic.py
│   │   └── planner.py
│   ├── training/
│   │   ├── collect.py
│   │   ├── pretrain.py
│   │   ├── infer_context.py
│   │   └── evaluate.py
│   ├── baselines/
│   │   ├── topology_only.py
│   │   ├── history_encoder.py
│   │   ├── continuous_descriptor.py
│   │   └── parameter_matched.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── storage.py
│   │   └── validation.py
│   └── analysis/
│       ├── aggregate.py
│       ├── bootstrap.py
│       └── plots.py
├── sim/assets/
│   ├── genkiarm_source.urdf
│   ├── genkiarm_calibrated.urdf
│   ├── arm.xml
│   └── meshes/
├── hardware/
│   ├── arm_spec.yaml
│   ├── joint_map.yaml
│   ├── safety_limits.yaml
│   └── calibration/
├── scripts/
│   ├── smoke_test.py
│   ├── run_g1.py
│   ├── run_g2.py
│   ├── run_real.py
│   └── reproduce_paper.py
├── tests/
│   ├── test_kinematics.py
│   ├── test_env_contract.py
│   ├── test_damage_model.py
│   ├── test_context_shapes.py
│   ├── test_data_schema.py
│   └── test_determinism.py
├── runs/                 # gitignored
├── datasets/             # gitignored; manifest tracked
├── checkpoints/          # gitignored; manifest tracked
├── results/              # aggregate CSV/JSON tracked when final
├── reports/
├── experiments/
├── reviews/
└── paper/
```

### 9.1 工具选择

- Python 3.11 优先；若选用的 Dreamer 实现不兼容，则固定 3.10；
- 依赖统一写入 `pyproject.toml` 和 lock file；
- `pytest`；
- `ruff`；
- 类型检查至少覆盖数据 schema 与 env protocol；
- YAML + dataclass/Pydantic，避免早期引入复杂配置框架；
- TensorBoard + CSV/JSONL 为默认日志；W&B 可选，不作为复现依赖。

### 9.2 最低测试

- MuJoCo XML 编译；
- J1–J5、URDF joint、舵机 ID 与动作索引一一对应，ID6 单独作为夹爪；
- 六维 joint mask/action/state shape 固定，夹爪开合通道不得混入；
- reset/step 1000 步无 NaN；
- lock joint 的动作屏蔽与状态变化符合配置；
- FK 与 MuJoCo site position 一致；
- 同 seed 的短 rollout 可重复；
- 数据写入后可无损读回；
- context shape 与梯度范围正确；
- evaluation 不更新模型；
- calibration targets 不出现在 evaluation split。

---

## 10. 数据、日志与实验治理

### 10.1 轨迹 schema

每条 episode 至少包含：

```text
episode_id
timestamp_ns
platform: sim | real
task_id
target_id
split: calibration | validation | evaluation
damage_id
joint_mask                 # length 5: J1...J5
lock_angle
observation
action_commanded           # 5-DoF arm action; gripper_open separate
action_applied
gripper_open_command       # nullable independent actuator
next_observation
reward
success
done
safety_flags
hardware_state
camera_frame_ref
config_hash
git_commit
seed
```

### 10.2 不可变性

- 原始轨迹只追加，不原地编辑；
- 清洗生成新 dataset version；
- 每个 dataset 有 manifest、样本数、hash、来源和排除原因；
- 最终结果引用 dataset version 与 commit；
- calibration/evaluation split 创建后冻结。

### 10.3 Run 命名

```text
{stage}_{task}_{damage}_{method}_k{K}_seed{seed}_{yyyymmdd-hhmm}
```

每个 run 保存：

- resolved config；
- stdout/stderr；
- metrics JSONL；
- checkpoint；
- environment/system info；
- git commit；
- wall-clock、GPU 型号、peak memory；
- exit status。

### 10.4 结果发布规则

- 论文表格只能从 `results/final/*.csv` 自动生成；
- 手工复制数字必须二次核对；
- failed runs 不删除，写入 exclusion ledger；
- exclusion rule 在查看 test 结果前确定；
- 图表保留 source data。

---

## 11. 硬件与安全 SOP

### 11.1 开机前

1. 检查结构件、螺丝、线缆和电源；
2. 确认工作区无人和无易碎物；
3. 载入对应安全配置；
4. 验证急停；
5. 低速回零；
6. 读取温度、电压、电流和位置；
7. 相机与机械臂坐标标定检查。

### 11.2 锁定前

- 断电或进入安全模式；
- 记录锁定关节与角度；
- 按固定机械步骤安装；
- 拍照记录；
- 手动小幅加载，确认微动范围；
- 不使用“继续加扭矩直到不动”的不可控方式。
- D2/D3/D4 分别使用经过验证的专用固定位置与夹具；不得把适用于某一中间关节的锁定力矩直接复制到其他关节；
- J1 底座、J5 腕部和夹爪故障不进入主实验，除非重新完成工作区、碰撞和末端姿态安全评审。

### 11.3 运行中

- 操作者始终在场；
- 先执行低幅安全动作；
- 任何软限位、通信超时、异常电流、异常温升触发 stop；
- 阈值必须来自数据手册或 G0 实测，禁止在计划中虚构固定数字；
- 每个 condition 之间进行冷却与外观检查。
- 对六个关节分别记录 commanded/applied position；锁定关节出现超出 G0 微动范围的位移时立即停止；
- 夹爪开合执行器与 J6 姿态命令独立限幅，避免通道映射错误造成夹持或碰撞风险。

### 11.4 结束后

- 保存完整日志；
- 记录急停、超限和人工干预；
- 检查舵机温度与松动；
- 生成 session summary；
- 不把发生人工干预的 episode 混入普通成功率。

---

## 12. 资源分配

| 资源 | 角色 | 不承担 |
|---|---|---|
| MacBook | 代码、测试、MuJoCo smoke、真机控制、分析、论文 | 长时全量预训练 |
| RTX 3070 | 小模型、单 cell、消融 smoke、回归测试 | 大 batch 全主表 |
| 云 RTX 4090 | WM pretraining、G1/G2 主实验 | 未通过 smoke 的调试任务 |
| 实体臂 | G0 测量、集成、G3 正式评估 | 没有安全限制的探索 |

### 12.1 预算明细

| 类别 | G0 | G1 | G2 | G3 | G4 | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| 工程/研究工时 | 16–24 | 40–60 | 60–90 | 30–45 | 30–45 | 176–264 |
| GPU-h | 0–5 | 30–60 | 60–100 | 8–16 | 0–5 | 98–186 |
| 真机小时 | 4–8 | 0–2 | 0 | 8–16 | 0–2 | 12–28 |
| 存储增量 | <5 GB | 20–40 GB | 40–150 GB | 15–35 GB | <10 GB | 75–240 GB |

### 12.2 预算审批点

- 单 run 超过预计时间 2 倍：暂停批量；
- G1 超过 60 GPU-h 仍无机制信号：强制评审；
- G2 预计超过 100 GPU-h：必须删除次要消融或重新批准；
- 真机发生一次严重安全事件：暂停，完成 root-cause report 后再恢复。

---

## 13. 时间表与关键路径

### 13.1 ICRA 六周倒排路线

| 周 | 日期 | 主任务 | 硬输出/降级规则 |
|---|---|---|---|
| W0 | 08-06–08-09 | J1–J5+夹爪映射、URDF 修复、FK、急停与测量模板 | joint map、URDF gap report、安全边界 |
| W1 | 08-10–08-16 | 5-DoF MuJoCo、可达域、Reach env、数据管线、intact/D3 baseline | G0 通过；健康 baseline 不稳则暂停方法扩展 |
| W2 | 08-17–08-23 | residual latent、factorized G1、3 seeds、D3 真机 pilot、方法稿 | G1 无信号则缩主张或 Pivot，不伪装结果 |
| W3 | 08-24–08-30 | D2/D3/D4、四基线、held-out 核心表；第二真机 condition；视频采集 | Push、K=2/10 和非核心消融可降级 |
| W4 | 08-31–09-06 | 核心消融、真机正式统计、主结果冻结、完整 8 页稿 | 09-04 冻结数字；未完成的强包项目删除 |
| W5 | 09-07–09-13 | 视频上传、failure analysis、内部审稿、引用/数字/匿名/PDF QA | 09-10 完成审计，09-12 完成预提交 |
| Submit | 09-14–09-15 | 最终检查、PaperPlaza 上传与回读 | 不在最后一小时首次上传 |

### 13.2 关键路径

```text
硬件测量
  -> 可达域与任务冻结
  -> MuJoCo 环境
  -> 健康/损坏 baseline
  -> factorized G1
  -> G1 Go
  -> held-out G2
  -> G2 Go
  -> 真机正式统计
  -> 论文结果冻结
  -> integrity/submission check
```

### 13.3 可并行任务

- 环境实现与文献监控；
- 仿真批量与真机 adapter 开发；
- G2 运行与图表脚本；
- 真机采样与论文方法重写；
- artifact QA 与投稿格式检查。

不能并行绕过的依赖：

- 没有 G0 不冻结 task；
- 没有 G1 不跑 G2；
- 没有稳定 G2 不做正式真机大样本；
- 没有冻结结果不写结论。

---

## 14. 第一阶段逐日任务

### Day 1

- 创建 `pyproject.toml`、`src/`、`tests/`；
- 固定 Python 版本选择流程；
- 写 `RobotEnv` protocol；
- 建立 `hardware/arm_spec.yaml` 与 `hardware/joint_map.yaml` 模板；
- 冻结 J1–J5、夹爪、URDF joint、舵机 ID、动作索引和功能角色；
- 保存当前 git 状态，不触碰现有未跟踪文件。

### Day 2

- 补齐 URDF mesh 路径，记录 inertial/collision/dynamics 缺口；
- 测量六关节连杆、TCP 与零位；
- 实现 FK；
- 写 FK 单元测试；
- 定义 D2/D3/D4；
- 草拟 safety limits。

### Day 3

- 构建最小 MJCF；
- 加载、reset、step；
- 校验 MuJoCo site 与 FK；
- 输出 intact reachability。

### Day 4

- 输出 D2/D3/D4 reachability；
- 生成共同目标集合；
- 决定 Pick 是否删除；
- 写 G0 中期报告。

### Day 5

- 真机位置响应与延迟测量；
- 自由状态回差测量；
- 急停测试；
- 修订安全配置。

### Day 6

- 安装锁定机构；
- 测量锁定角微动、摩擦代理量和重复性；
- 拍照与保存原始数据；
- 如出现风险立即停止。

### Day 7

- 完成 G0 gate review；
- 冻结 Reach target split；
- 更新 MuJoCo parameter ranges；
- 决定是否进入 G1。

### Day 8–10

- 实现 Reach reward/success；
- 轨迹 schema 与 validation；
- topology-only policy/WM baseline；
- 1000-step 与短训练 smoke。

### Day 11–14

- residual latent optimization；
- factorized conditioning；
- frozen actor/MPC；
- 第一个 D3、K=0/1/5、seed 0 对照；
- 根据真实 wall-clock 更新 G1 预算。

---

## 15. 风险登记表

| ID | 风险 | 概率 | 影响 | 早期信号 | 缓解 | 触发决策 |
|---|---|---|---|---|---|---|
| R0 | URDF 与真机不一致 | 高 | 致命 | FK/TCP、轴向或零位明显偏差 | joint map；10+ 姿态校验；校准后再转 MJCF | 阻塞 G0 |
| R1 | 锁定后任务不可达 | 中 | 致命 | IK/采样可达率低 | 使用共同可达域；删 Pick | Stop/缩范围 |
| R2 | factorization 无收益 | 中高 | 致命 | G1≈topology-only | 转 benchmark/zero-shot | Pivot A |
| R3 | actor BC 才有效 | 中 | 高 | actor frozen 无提升 | 转 few-shot imitation | Pivot B |
| R4 | history encoder 更强 | 中 | 高 | 短历史已识别全部变化 | 转未知损坏 diagnosis | Pivot |
| R5 | WM 训练不稳定 | 中高 | 高 | loss/return 高方差 | 小 RSSM；MPC；复用可靠实现 | 架构降级 |
| R6 | sim-to-real gap 过大 | 高 | 高 | 真机 prediction error 不降 | 扩 residual ranges；实测 actuator model | 降低主张 |
| R7 | 硬件损坏 | 低中 | 高 | 温升、电流、松动 | 安全边界、备件、监督运行 | 暂停 |
| R8 | 计算超预算 | 中 | 中 | run >2×预计 | early stopping、缩 seed/cell | 预算复审 |
| R9 | competitor 抢先 | 中 | 高 | 2026 新预印本重合 | 月度监控；调整 novelty delta | idea 复评 |
| R10 | ICRA 截止诱发低质量提交 | 高 | 高 | 9/5 仍无冻结结果 | 硬退出门 | 转 RSS/CoRL |
| R11 | 单平台证据不足 | 高 | 中高 | reviewer 认为 demo-only | 多 condition、多日、开源 protocol | 系统贡献增强 |
| R12 | 统计功效不足 | 中 | 高 | CI 极宽 | 增 evaluation episodes；报告效应量 | 克制结论 |
| R13 | 数据泄漏 | 中 | 致命 | calibration/eval targets 重合 | split hash、自动检查 | 重跑 |
| R14 | 论文旧叙事残留 | 高 | 高 | binary>continuous 仍出现 | G2 后全稿重写而非局部替换 | integrity audit |

---

## 16. 论文与证据同步计划

### G0 后可写

- 平台与故障定义；
- 可达域构造；
- 安全与测量协议；
- 任务范围。

### G1 后可写

- DFWM 方法；
- context factorization；
- deployment inference；
- G1 作为内部证据，不一定进入最终主表。

### G2 后可写

- 实验设置；
- baseline 公平性；
- 主结果、消融、held-out composition；
- failure cases；
- compute。

### G3 后可写

- 真实平台结果；
- sim-to-real 分析；
- 安全与 wall-clock；
- 局限。

### 必须整体删除或重写的旧内容

- binary vs continuous 的表达能力论证；
- random mask “保证任意损坏 in-distribution”的泛化表述；
- token + actor-head 联合微调作为核心机制；
- 预期 60%/80%、15 points 胜幅；
- 500 GPU-h 与碳排的未实测数字；
- RSS 2026 仍可投稿的时间线。

### 论文 owner 顺序

1. `ccf-experiment-designer`：G0 结果出来后冻结完整实验协议；
2. `ccf-paper-writer`：G1 通过后重写问题和方法；
3. `ccf-visual-composer`：G2/G3 真实数字冻结后生成图表；
4. `ccf-integrity-auditor`：核查 claim、数字、引用和表图；
5. `ccf-paper-reviewer`：模拟主会评审；
6. `ccf-submission-checker`：官方 CFP 发布或投稿前核验。

---

## 17. 项目管理节奏

### 每日

- 今天完成了什么；
- 新生成哪些 artifact；
- 哪个假设被支持/削弱；
- GPU/真机消耗；
- blocker；
- 明天唯一最重要任务。

### 每周

```text
Week:
Gate:
Completed:
Evidence:
Failed/Excluded runs:
Budget used:
Risks changed:
Decision:
Next week:
```

### 每个阶段门

必须留下：

- 输入版本；
- 运行列表；
- pass/fail 证据；
- 预算实际值；
- Go/Pivot/Stop 决策；
- 决策人和日期；
- 下一阶段范围。

---

## 18. Artifact 清单

### G0

- [ ] arm spec
- [ ] safety limits
- [ ] calibration raw data
- [ ] reachability split
- [ ] MuJoCo XML
- [ ] feasibility report

### G1

- [ ] reproducible environment
- [ ] data generator
- [ ] topology encoder
- [ ] residual context
- [ ] conditional WM
- [ ] frozen control
- [ ] four-method pilot
- [ ] gate report

### G2

- [ ] immutable configs
- [ ] minimum 3-seed main table；资源允许时扩展到 5 seeds
- [ ] held-out composition
- [ ] recovery curve
- [ ] ablations
- [ ] robustness
- [ ] failure analysis
- [ ] compute report

### G3

- [ ] real raw trajectories
- [ ] safety ledger
- [ ] two fault conditions
- [ ] 最低包 ≥20 episodes/condition；强包 ≥30 episodes/condition
- [ ] videos
- [ ] sim-to-real analysis

### G4

- [ ] revised manuscript
- [ ] source-data figures
- [ ] references verified
- [ ] artifact README
- [ ] checkpoint manifests
- [ ] integrity audit
- [ ] simulated review
- [ ] official submission check

---

## 19. `ccfa.yaml` 建议更新

按 orchestrator 规范，本轮不自动改写 `ccfa.yaml`。建议在用户明确批准项目状态迁移后：

```yaml
project:
  title: 六自由度低成本机械臂关节锁定后的损坏因子化世界模型与少样本安全恢复
  revised: 2026-08-06

target_venue:
  primary: ICRA-2027
  fallback:
    - RSS-2027
    - CoRL-2027
  submission_deadline: 2026-09-15T23:59:00-08:00
  rule_verified: 2026-08-06
  deadline_policy: verify-official-again-before-submission

stage: v4-6dof-plan-ready-g0-started

claims:
  central_claim:
    statement: 已知离散损坏拓扑与未知连续残余动力学的因子化，可降低低成本机械臂关节锁定后的真实校准数据需求
    status: needs-g1-mechanism-evidence
  sub_claims:
    - statement: actor 与 world model 冻结时，residual context inference 仍带来恢复
      status: needs-g1
    - statement: factorization 可泛化到 held-out topology-physics 组合
      status: needs-g2
    - statement: 方法在至少两个真实锁定条件下可重复
      status: needs-g3

artifacts:
  - reviews/idea-review-robotics-topvenue-20260730.md
  - PROJECT-PLAN-V4.md
```

旧实验 E1/E2/ABL/ROB 不应直接删除；建议标记为 `superseded-by-v4-gates`，保留历史审计。

---

## 20. 当前 Gate 决策与下一 owner

**当前阶段**：G0 完整复审已补齐校准 URDF、缺口报告、位置/姿态可达域图、动态汇总和任务裁决；在 position-only Reach 范围内带偏差说明通过。G1/Pivot 控制复核仍为 No-Go。
**当前 gate**：暂停 G2；G0 不再阻塞，当前阻塞项是 G1 学习式控制仅 1/3 seeds 成功。
**下一执行 owner**：

1. 项目初始化与工程实现；
2. G0 测量；
3. `ccf-experiment-designer` 在测量结果之后冻结完整实验协议。

**G0 已获得输入**：

- `genkiarm.urdf`：确认 7 links、6 revolute joints 及名义轴/偏移；
- 真机确认：J1 底座、J2 大臂、J3 小臂、J4 腕俯仰、J5 腕旋转、ID6 夹爪开合。

**G0 仍缺输入**：

- URDF 引用的 `AAA.stl`–`FFF.stl` 网格，或可替代的 CAD/碰撞几何；
- 舵机 ID、零位与方向；
- 控制接口地址/调用方式；
- 锁定结构照片或说明；
- 相机型号与安装方式；
- 可接受的真机实验时间。

在这些输入未齐前，可以完成仓库初始化、MuJoCo 骨架、FK、schema、测试与安全模板，但不能诚实冻结真实物理参数和正式任务分布。

---

## 21. V5 完成标准

本计划本身完成不等于项目完成。项目完成的最终定义是：

- 研究问题与方法无内部矛盾；
- G0–G4 留有可审计 artifact；
- 结果支持的主张与论文一致；
- 失败结果和适用边界被披露；
- 真实系统安全、统计和成本透明；
- 官方会议规则在提交前重新核验；
- 不依赖预期数字、不可复现脚本或单次展示。
