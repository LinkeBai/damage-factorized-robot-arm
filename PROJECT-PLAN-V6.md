# Project Plan V6 — Robust Zero-Shot Structured Dynamics

## 2026-08-29 实际模型与双 eye-to-hand 资产复核

重新确认后，论文与后续实验的权威工程目录为本完整项目，而非 `Documents/ChatGPT/New project/robotarm` 中的简化机制原型。后者新增的恒定动作/序列 soft-regret 实验只可作为探索诊断：其最好 D2/D4 闭环终点误差改善为9.27%，仍低于10%门槛，不迁入论文主表，也不改变本计划中3.2--3.4/5的评分。

本项目已包含学长要求的两套实际仿真资产：校准 GenkiArm Push，以及官方 MuJoCo Menagerie Franka Panda + 双指夹爪 + 方块的 Push/Grasp wrapper；两套模型均定义两个固定在环境中的 `eye_to_hand_left/right` 相机，而非手眼相机。资产回归测试当前 `9 passed`。

可视化审计发现并修复两个会污染实验的问题：

1. GenkiArm 与 Panda 的右侧相机 `xyaxes` 镜像符号错误，原图完全朝向场景外；已修复并确认左右视角均覆盖机械臂、桌面和任务区域。
2. Panda 上游 `home` keyframe 在 wrapper 新增自由方块后会把方块补零到世界原点；已新增完整 `task_home`（包含方块 `[0.50,0,0.025]`）并让 task environment 使用它，同时增加回归测试。

四视图证据位于 `reports/actual-model-eye-to-hand/dual_eye_to_hand_actual_models.png`。该结果只证明实验资产与视觉来源正确，不等于视觉鲁棒性、Grasp性能或跨臂泛化已经成立。

### 操作空间无量纲干预诊断

在确认 raw history 的跨臂不对称后，新增了不依赖网络结构的解析候选：以锁定前后 `J_f M_ff^{-1}τ_f` 为动作响应，用操作空间 mobility、控制能量和当前接触坐标系白化，消除DoF、关节坐标与执行器尺度差异。闭式 Ridge 诊断在 seeds 7/17/27 上获得 pooled RMSE 改善 `23.65%/16.05%/19.28%`，其中2/3 splits同时改善GenkiArm与Panda，说明该归一化确实缓解跨embodiment预测不对称。

但动作排序 Spearman 变化仅 `-0.0289/+0.0518/+0.0122`，全部低于冻结的 `+0.10`，seed27 top-1 regret还略微变差，联合Gate为 **0/3 No-Go**。正式报告见 `reports/ipwm-operational-invariant-effect-diagnostic-20260829.md`。该结果只能作为机制诊断，不得升级为控制相关核心创新，也不批准Push+Grasp五seed矩阵。

## 2026-08-29 论文主张—证据审计（当前最高优先级）

正式审计见 `reports/paper-claim-evidence-audit-20260829.md`。当前论文没有虚构
Grasp、跨机械臂、闭环或真机成功；它支持的是单机械臂 planar Push 上较窄的
SI-IPWM 状态隔离主张。按冻结 ICRA/CCFA 证据合同，当前综合评分仍为
**3.2--3.4/5（weak reject）**，不能因诚实报告 No-Go 而上调到 4.0。

以下证据边界立即冻结：

- solver-native/CFPO deployable Gate 为 No-Go，不能作为已验证核心创新；
- 跨结构 robot-transition Gate 仅 2/3 seeds 通过，且两种结构均参与训练；
- 跨结构 object/contact Gate 为 0/3 seeds No-Go，禁止声称故障影响已成功传播到对象；
- GenkiArm 冻结迁移只有部分 open-loop 正证据，不能声称部署或控制成功；
- variable-DoF 只证明统一接口，不证明性能泛化；
- Push + Grasp/Pick、双 eye-to-hand 视觉鲁棒性和真机均尚无完整证据链。

因此，在出现新的、可证伪且非“继续堆组件”的核心假设前，不启动双任务五 seed
确认矩阵，也不把上述局部结果拼接成完整系统贡献。下一允许动作仅是提出并冻结一个
同时作用于 object/contact prediction 与 action ranking 的最小机制 Gate；它必须在
GenkiArm 与 Panda 上，以相同可观测量、参数匹配基线和至少 3 个开发 seed 验证，
至少 2/3 seeds 同向通过且无按 seed 例外，才允许扩展到 Push + Grasp/Pick。

### 2026-08-29 contact action-effect 最小机制 Gate

按照上述规则，冻结并执行了“解析故障投影后的末端候选动作响应 + 低秩接触传播
算子”Gate，正式报告为 `reports/ipwm-contact-action-effect-gate-20260829.md`。
数据同时来自校准 GenkiArm 与官方 Panda：每臂 80 个当前接触前缀、每前缀 6 个
候选动作，共 2,880 条严格同状态 counterfactual rows；以 prefix 分组，两个机械臂
分别 held out 中间关节。结构模型与相同输入 flat MLP 参数差 0.53%。

结果为 **0/3 seeds No-Go**：pooled RMSE 相对改善为 -25.09%、+4.75%、+6.32%，
没有一个 seed 同时改善两种机械臂；Spearman 动作排序变化为 -0.0119、-0.2631、
-0.2786，top-1 regret 三个 seed 全部更差。按预注册规则停止，不调整 rank、hidden、
loss、feature、动作扰动、阈值或 seed。该路线不能作为论文正贡献，也不批准双臂
Push + Grasp/Pick 五 seed 矩阵。

### 2026-08-29 多步动作可识别性诊断

在不训练或重调失败算子的前提下，对相同前缀/故障/候选动作执行 H1/H5/H10 分支，
报告见 `reports/contact-action-observability-diagnostic-20260829.md`。H1 对 GenkiArm
的最佳动作中位差只有约 0.010 mm，不足以代表控制尺度；H10 时两臂候选动作的
中位 XY 范围均约 2.4 mm，末端响应差异与物体响应差异的 Spearman 分别为 0.630
和 0.404，说明多步尺度存在非平凡动作信号。

与此同时，H10 最终接触保持率仅 GenkiArm 60.7%、Panda 45.6%。因此剩余唯一允许
查新的机制假设不是“再换一个回归 head”，而是把多步故障反事实分解为显式的
contact-mode survival 与 mode-conditioned cumulative response；解析锁定投影和状态
隔离继续承担硬约束与 free-joint 安全边界。该假设必须先完成 hybrid/switching
world-model 查新并冻结公平 Gate，不能把本诊断计作论文正证据。

### 2026-08-29 hybrid fault-counterfactual 查新与 Gate 冻结

查新报告为 `reports/hybrid-fault-counterfactual-novelty-audit-20260829.md`。显式
contact mode、switching dynamics、contact/continuous 联合学习和跨 embodiment
contact representation 均有直接近邻，因此“接触分类器 + 两个响应 head”只有
3.0--3.3/5 的组件组合创新度。当前仅可检验的差异化整体是：未见结构故障干预、
解析可行流形、受保护 carrier、故障条件化的 contact survival、模式内累计响应，
并以动作排序和 regret 判定。即使完整通过，现阶段创新度预期也仅冻结为
3.6--3.9/5，不预称 4.5/5 或 5/5。

最小 Gate 已在 `config/experiment/ipwm_fault_hybrid_counterfactual_gate_v1.yaml`
冻结：H10、双臂、held-out 中间锁、prefix 分组、3 个开发 seeds；同时对比参数匹配
flat multi-task 与 non-mixture 强基线，并联合要求 contact calibration、两臂 object
RMSE、Spearman 与 top-1 regret 通过。

该 Gate 已运行并为 **0/3 No-Go**，正式报告见
`reports/ipwm-fault-hybrid-counterfactual-gate-20260829.md`。seed 7/17/27 相对最强
基线的 RMSE 改善分别为 -5.21%、+4.54%、+12.93%，Spearman 改善仅 +0.0179、
-0.0286、+0.0223；contact balanced accuracy 为 0.613/0.659/0.633，Brier 为
0.346/0.345/0.383。即使 seed 27 的 RMSE 与 regret 通过，它仍同时失败于 mode
calibration 和动作排序效应阈值，不能算 partial Go。按冻结规则停止，不调整模式、
H10、loss weight、head、hidden、feature、阈值或 seed。

### 2026-08-29 active-contact 少样本可识别性最终诊断

正式报告见 `reports/ipwm-active-contact-identifiability-20260829.md`。有效数据集为
`runs/ipwm_active_contact_identifiability_v1/dataset_v2_seed20260829.npz`：双臂各
7,200 行、共 14,400 行，覆盖 K8 probe、H10 candidate、五物理 profile、三故障和
六候选动作，锁定误差为零。旧 `dataset_seed20260829.npz` 因历史字段覆盖当前字段
而作废，仅保留审计，禁止训练或制表。

ordered history 相对 current-only 在三个 grouped split 上的 pooled RMSE 改善为
35.02%/38.43%/41.28%，Spearman 提升 +0.384/+0.556/+0.454，且均明显优于等维度
permuted-history 控制；说明历史确有局部任务响应信息。但三个 split 均只改善 Panda
并退化 GenkiArm，按“两种结构必须同时受益”规则为 **0/3 No-Go**。robot-specific
Ridge 信息上界呈相反不对称：history 改善 Genki，Panda 则 current-only 更好；物理
参数解码也全部差于常数基线。因此不能声称统一跨结构物理 context 已可识别。

按预注册停止规则，当前不得再后验加入 graph encoder、GRU、contrastive loss、
per-robot normalization 或改变 probe。核心机制未通过，故不启动 Push + Grasp/Pick
五 seed 确认矩阵。当前仿真论文仍约 3.2--3.4/5；继续达到 4+/5 需要独立的新理论
原则或用户批准改变论文范围，不能靠包装或继续堆组件。

**项目**：低成本机械臂关节锁定后的稳健零样本结构化动力学
**版本日期**：2026-08-28（近期工作汇总版）
**规划模式**：standard  
**规划基线**：本文件是后续执行的最新基线；旧版计划和失败实验保留为审计记录  
**当前状态**：论文核心重新冻结为 IPWM 的“解析约束干预 + 受影响路径上的选择性世界模型修正”。开环预测、约束保持和状态隔离已有部分正证据，但预测到闭环控制的稳定迁移尚未成立；free-joint 与完整闭环仍存在 No-Go。当前先以理论和仿真达到 ICRA 4.0/5 为目标，真机实验不计入当前达标条件。
**证据约束**：实测结果均指向可追溯 artifact；未来时间、GPU-h、工时和阈值仍属于项目管理估计

> ## 2026-08-28 给学长的近期工作汇总（当前权威基线）
>
> 本节优先于下方 2026-08-20/21 的历史流水账；旧内容仅作为失败路线和决策过程的审计记录。
>
> ### 1. 当前客观结论与目标
>
> - 按 ICRA 2027 与现有 CCFA 审查口径，当前综合评分约 **3.2--3.4/5**（Weak Reject 至 Borderline）；闭环控制整体仍为 No-Go 时不得评为 4.0/5。
> - 优势：解析锁定约束、结构化状态隔离、五 seed 预测证据、失败披露和实验治理。硬伤：闭环收益未成立、IPWM 相对外置 ranker 的独立贡献未证、当前无正式真机任务结果。
> - 当前目标：**保持 IPWM 核心创新不变，先仅依靠理论与仿真形成完整、可复现、无数据泄漏的证据链，使论文客观达到至少 4.0/5；九月真机后续单独加证据。**
>
> ### 2. 不变的核心创新
>
> > 对未见执行器锁定故障，将动力学变化分解为可解析的约束干预，以及仅沿受影响运动链/接触路径学习的支持集外反事实动力学修正。
>
> 1. 已知锁定坐标由解析投影严格满足位置与速度约束；
> 2. IPWM 负责自由关节、末端和物体状态的多步滚动、物理上下文及候选动作反事实预测；
> 3. 选择性干预只修正故障可能传播到的状态路径，避免全局残差污染未受影响状态；
> 4. 风险调度、MPC 或 ranker 只能消费 IPWM 信息，不能取代 IPWM 成为实际方法主体；
> 5. 最终贡献必须归因于 IPWM，而不是容量、终点标签、风险阈值或 MuJoCo 特权状态。
>
> 当前判断：机制逻辑成立且有创新内核，但目前只对约束保持、开环预测和状态隔离有部分证据；“选择性干预能否恢复任务所需的反事实动作响应”仍是待验证核心命题。
>
> ### 3. 相对 2026-08-24 最初审稿意见的改进
>
> | 最初问题 | 已完成或冻结的改进 | 当前裁决 |
> |---|---|---|
> | 论文与 Plan 脱节 | 冻结 ICRA 证据合同、配置、split、Gate 和机器可读结果要求 | 新实验按此治理；旧 18/18 数字不能自动继承 |
> | 3 seeds、统计不足 | 预测扩到 5 seeds；采用 seed-level/hierarchical paired bootstrap | 新闭环仍需五 seed |
> | 全正、没有失败域 | 保留 free-joint、接触传播和闭环 No-Go | 真实性提高，但失败本身不加性能分 |
> | 2% 阈值疑似后验 | 新实验在读取 evaluation 前冻结门槛和三段 split | 旧阈值问题不追溯性洗白 |
> | 组件堆叠、因果链不清 | 主线收缩为“解析约束 + 选择性干预”，停止多个无效分支 | 仍需决定性归因消融 |
> | 控制收益未证 | 新增动作排序和完整终点诊断，确认平均 RMSE 与控制排序目标错配 | 尚未转为 Go |
> | 真机表为空 | 不再把 G0 校准、安全记录或演示计为任务结果 | 九月另按独立重复协议补充 |
> | 视觉来源错误 | 第二视觉源纠正为固定水平外部摄像头；计划使用两个 eye-to-hand 摄像头 | 方法图和文字必须统一 |
>
> ### 4. 近期控制诊断与 No-Go
>
> - carrier 和 SI-IPWM 均由原 IPWM checkpoint 构建；当前工作没有绕过原世界模型。
> - 直接 rerank、接触门控、线性 ranker 与 H10 神经 ranker 在部分离线分支排序上有信号，但完整闭环发生目标回归，均不得写成控制成功。
> - 使用真实 MuJoCo 未来状态继续生成动作的完整终点数据只属诊断上界，包含部署不可得信息，已判为 **privileged / invalid for deployable evidence**，不得计分。
> - 修正为 deployable one-shot CEM 后，v3 validation Spearman=`0.482`，低于冻结门槛 `0.5`；平均终点改善约 `4.825%`，故在正式 evaluation 前判 No-Go。
> - v4 正扩展接触状态几何覆盖；数据采集与训练尚未完成，结果不得提前计分。
> - free-joint 失败表明解析投影虽保证锁定坐标，却不会自动恢复约束改变后的耦合动力学；应作为机制边界报告。
>
> ### 5. IPWM 不可替代性的强制消融
>
> 在相同数据、终点监督、候选动作、容量、规划预算和 split 下比较：
>
> 1. carrier MPC；
> 2. carrier + 不含 SI-IPWM 特征的 ranker；
> 3. carrier + SI-IPWM 反事实特征 ranker；
> 4. SI-IPWM 直接 MPC；
> 5. 普通世界模型、仅解析投影、投影 + 全局同容量残差、投影 + IPWM 选择性干预。
>
> 只有第 3 项稳定优于第 2 项，且选择性 IPWM 稳定优于全局同容量残差，才能把提升归因于 IPWM；否则 ranker 只能作为负面诊断。
>
> ### 6. 理论与仿真达到 4.0/5 的冻结门槛
>
> 1. 理论：零违例性质、选择性干预路径支持，以及动作排序误差与 MPC regret/任务损失的联系；
> 2. 反事实预测：未见关节、角度、目标和物理组合上优于普通模型、仅投影和全局残差；
> 3. 闭环：至少 5 training seeds、多目标、held-out 故障，paired/hierarchical bootstrap 95% CI 不跨零；
> 4. 鲁棒性：冻结模型测试双外部摄像头噪声、遮挡、延迟、外参误差及摩擦、质量、执行器延迟；
> 5. 失败边界：保留 free-joint/contact-sensitive regime、计算开销和失败案例。
>
> 只有开环提升而闭环 No-Go 时，综合评分上限约 `3.8/5`；IPWM 独立贡献、五 seed 闭环和组合泛化同时成立，才允许评为 `4.0--4.2/5`。
>
> ### 7. 强创新候选（尚未并入已验证方法）
>
> 提出 **Causal Fault-Propagation Operator IPWM（CFPO-IPWM）** 作为受控候选：不学习任意状态残差，而学习 `do(joint locked)` 对“动作到状态变化”映射的局部响应算子，并沿运动链和接触图传播。
>
> 仅当小规模预注册实验满足：动作排序相对原 IPWM 改善至少 20%、未见故障闭环终点误差改善至少 10%、至少 2/3 seeds 方向一致、无路径约束同容量算子不能获得相同提升，才升级为主方法；否则停止。
>
> **2026-08-28 裁决：NO-GO。** 严格完整前缀的三 seed 诊断证明锁定反力可解释
> free-joint 响应，且物体响应必须经接触约束传播；但仅用部署可得状态/动作/连续故障位置，
> 在 j2+j4 训练、未见 j3 + 独立 episode 测试时，共享路径传播模型只在 1/3 seeds 优于
> 参数匹配无结构 MLP，另外两 seed 分别恶化约 22.8% 与 87.8%，且 R2 全为负。因此按
> 冻结规则停止 CFPO/constraint-response 学习路线，不搜索网络宽度、不追加 gate/attention，
> 不进入动作排序或闭环。solver 分解只能作为机制诊断与失败边界，不能升级为论文核心创新。
>
> ### 8. 九月真机计划（不计入当前 4.0/5 门槛）
>
> - 一个夹爪机械臂、一个方块、两个固定 eye-to-hand 摄像头；第二视觉源为手在眼外的水平摄像头，不是手眼视觉。
> - 低速、短轨迹、固定夹爪开合的桌面 Push；方块每回合独立复位。
> - intact + 两个安全锁定条件，至少 3 个目标；每故障条件至少 20 个独立 evaluation episodes。
> - calibration、调参与 evaluation 分离，保存视频、逐回合 CSV、安全事件和标定记录。
>
> ### 9. 下一步执行顺序
>
> 1. 完成 v4 deployable 数据与验证审计；未过 validation 门则停止 ranker 路线；
> 2. 执行有/无 IPWM 特征的公平消融；
> 3. IPWM 独立贡献成立后，扩到五 seed 闭环、held-out 组合与鲁棒性矩阵；
> 4. 同步完成理论命题、方法因果图、Plan/论文/结果文件一致性修订；
> 5. 仅在冻结量表复审至少 4.0/5 后宣布理论与仿真阶段完成。
>
> ### 10. 2026-08-28 学长建议转化为冻结执行要求
>
> 学长的最新意见优先于此前“单机械臂、单Push任务即可完成仿真证据”的收缩方案。
> 当前执行必须同时覆盖以下两条主线，不能二选一：
>
> 1. **跨机械臂结构泛化**：在现有5-DoF低成本机械臂之外，增加至少一种公开、结构不同且
>    可复现的机械臂模型。冻结训练/适配协议后，报告未见锁定关节、锁定角度及结构差异下的
>    预测、约束、动作排序和闭环结果。第二机械臂不得只作演示视频。
> 2. **跨任务泛化**：保留Push，并增加Grasp或Pick-and-Place。抓取任务必须具有可重复的
>    成功判据、共同可达域和独立evaluation episodes；若现有夹爪结构不能稳定完成，允许重新
>    设计并3D打印夹爪，但不能用抓取器失败替代方法评估。
>
> 同时新增三项强制工作：
>
> - **差异化查新**：建立逐项近邻矩阵，明确IPWM相对因果世界模型、动作干预一致性、
>   约束/混合动力学、关节故障容错和图传播方法的新性质。未完成查新前禁止宣称“首个”。
> - **仿真可视化**：从现在开始保存相同初态/动作下的intact、baseline、IPWM轨迹，包含
>   锁定影响传播、接触事件、候选动作预测与真实结果。正式图必须由冻结源数据生成。
> - **真机定位**：由于现有真机性能有限，真机只承担简单、低速、可重复的核心机制验证；
>   泛化主证据由跨机械臂和跨任务仿真承担，不能依赖几段真机展示补足。
>
> 新的证据升级顺序为：差异化与机制Gate → 3-seed小实验 → 第二机械臂 + Grasp smoke →
> 5-seed跨结构/跨任务正式矩阵 → 可视化与论文 → 简单真机验证。任何阶段No-Go均如实归档，
> 不通过继续增加模块、改阈值或选择性删除失败条件制造正结果。
>
> ### 11. 实际模型与简化模型的冻结分工
>
> - `sim/assets/arm_push.xml` 只用于单元测试、机制调试和低成本 No-Go，不进入跨任务、
>   跨结构主表，也不得称为实际机械臂模型。
> - GenkiArm Push 主证据使用 `sim/assets/genkiarm_push.xml`：关节轴、限位、连杆/TCP
>   偏置继承实物标定文件，外观继承原始 CAD；接触使用与标定 URDF 一致的 primitive
>   collision proxies。尚未辨识的惯量、阻尼、驱动和摩擦必须标为 provisional，并通过
>   冻结物理随机化覆盖，禁止称为完整 digital twin。
> - Grasp 必须在同一标定机械臂上补齐实际左右指爪、开合关节、行程、指尖碰撞面和 servo-6
>   映射后才可进入主表。当前整体 `GG.stl` 与固定 tool collision 不构成可验证抓取模型；
>   在缺少实测尺寸时不得虚构夹爪或用理想 weld 抓取作为正式结果。
> - 第二机械臂必须采用公开、完整、可复现的动力学与夹爪模型；其作用是跨结构验证，不能
>   替代 GenkiArm 上的实际标定模型结果。
> - 两个固定相机均为 eye-to-hand。主视觉结果必须使用这两个外部视角及冻结噪声/遮挡/
>   延迟/外参误差；overview 相机仅用于论文可视化，不是方法输入。
>
> **2026-08-28 冻结 checkpoint 零样本迁移审计：部分通过，非正式 Go。** seed 27/37/47
> 原始 selective IPWM 在标定 GenkiArm 上改善 7/9 个 horizon 单元，三 seed 平均改善
> `10.82%`；但 H50 在 seed 27/37 分别恶化 `10.83%/3.61%`。冻结 support router 在前
> 两 seed 回退，仅 seed 47 启用，因此部署路径为 3/9 改善、6/9 平局，95% 区间下界为
> `0`。free-state 变化和锁定违例均为零。该结果只支持“安全预测隔离可迁移”的探索性
> 结论，不证明稳定部署收益、跨结构泛化或闭环控制，也不提高当前评分至 4.0。
>
> **2026-08-28 双臂跨结构裁决：robot-transition 2/3 PASS；object/contact 0/3
> NO-GO。** 同一共享图机制在完整 GenkiArm 5-DoF 与 Panda 7-DoF 上，对各自未见中间
> 锁定关节的自由关节预测相对参数匹配 MLP 在 seed 7/27 两臂均改善，seed 17 出现
> GenkiArm 回退，因此只构成小规模窄正证据。进一步使用严格同前缀接触反事实、相同部署
> 输入及 0.24% 参数量差的对象传播比较时，结构模型在 seed 7/17/27 的 pooled RMSE
> 分别恶化 28.0%、63.6%、13.2%，0/3 通过。按预注册规则停止当前可变结构 object head，
> 不进入双臂 Push/Grasp 五 seed 主实验，不增加 contact gate/attention 或改 loss。论文只能
> 报告共享链对 robot transition 的有限证据及其无法迁移到接触对象的失败边界。

> ## 2026-08-21 可转发执行摘要（历史状态，已被 2026-08-28 基线取代）
>
> ### 1. 当前核心方法
>
> 新候选方法暂称 **Dual-Expert Damage World Model（DE-DWM）**。它不是继续给
> DFWM 增加 latent/contact head，而是把已有正结果组合成不可旁路的 product-space
> 分工：
>
> - **Structural expert**：FT-GWM K1，保留锁定连杆的固定 SE(3) 几何，仅预测
>   joint state，并解析保证锁定关节位置/速度约束。
> - **Predictive expert**：ordinary constant-condition deep ensemble，负责
>   object state 与经验不确定性。
> - **Product-space fusion**：下一状态的 joint 来自 structural expert，object
>   来自 predictive expert；第一版冻结两个专家，不训练额外 gate。
> - **待验证核心量**：两个异构专家在 joint 子空间的分歧
>   `u_cross = RMSE(joint_data_expert, joint_structural_expert)`。目标是检测普通
>   ensemble 成员可能共同犯下、因而无法被内部 disagreement 暴露的结构错误。
>
> ### 2. Q0-A 融合保真结果
>
> - Q0-A 使用 leave-one-joint-out 冻结协议、相同训练/评估轨迹、seed 7/17。
>   主域 D3 mixed composition 的 object RMSE 分别从 `0.3103/0.1467` 变为
>   `0.3036/0.1438`（改善 `2.15%/1.98%`）；free-arm RMSE 分别改善
>   `53.86%/41.08%`；constraint violation 均为 `0`。两 seed 均通过 object
>   回退不超过 2%、free-arm 回退不超过 5%、violation 不超过 `1e-7` 的门槛。
> - 当前结论仅为 **Q0-A TWO-SEED PASS**：证明冻结异构专家能够组合且保持预测
>   保真。尚未证明 cross-expert discrepancy 提供独立风险信息，也未证明控制收益；
>   不得把 Q0-A 写成风险感知或控制性能已经成立。
> - 权威报告：`reports/g2-dual-expert-gate-q0a-20260821.md`。
>
> ### 3. 旧结论如何串联到新方法
>
> | 已有证据 | 对 DE-DWM 的约束 |
> |---|---|
> | ordinary ensemble 相对参数匹配单模型改善 `30.74%`，95% CI `[15.06%, 42.62%]`，5/5 seeds | 保留为 predictive expert |
> | structured vs ordinary ensemble 仅改善 `2.47%` 且 CI 跨零 | 不再把 topology conditioning 本身作为预测创新 |
> | 50% coverage 下 selective RMSE 下降约 `50.50%` | 保留 ensemble uncertainty，但固定深度重新校准 |
> | FT-GWM K0 PASS、K1 two-seed provisional PASS | 保留为 structural expert |
> | FT-GWM K2、FTC-WM L、hybrid-contact M、multi-contact N 均 No-Go | structural branch 不再学习 object/contact |
> | Guarded MPC 的统计区间跨零 | Q0-B 前不做控制收益主张 |
>
> 因此旧工作没有作废：它们构成了“预测专家擅长 object、结构专家擅长约束，任何
> 单一专家都不足”的证据链；但旧数字只能支持设计动机，不能替代 DE-DWM 的新实验。
>
> ### 4. MuJoCo Warp 加速试验
>
> 本机环境为 MuJoCo `3.11.0`、Warp `1.16.0`、RTX 4060 Laptop GPU。
> raw physics benchmark 结果为：32 worlds 时 CPU 约 `267k steps/s`、Warp 约
> `90k steps/s`；256 worlds 时 CPU 约 `177k steps/s`、Warp 约
> `678k steps/s`，Warp 约 `3.8x`。100 步一致性测试的 qpos/qvel RMSE 约为
> `8.3e-8/1.0e-7`。
>
> 结论：Warp 在数百环境的大 batch 下有价值，但当前每次约 24 条训练轨迹时反而
> 不能加速；首次 JIT 还需约 31 秒。因此现阶段保留 CPU MuJoCo 冻结数据协议，待
> Q0-B 需要数百/数千条校准轨迹时再接入 Warp。当前训练加速优先级是 FT-GWM 边
> 传播张量化、FK 缓存和 rollout 编译。
>
> ### 5. 下一步冻结决策
>
> 下一唯一方法实验是 **Q0-B fixed-depth conditional-risk gate**：
>
> 1. 在每个固定 rollout depth 分别计算 ensemble disagreement、`u_cross` 和真实误差；
> 2. 检验 `u_cross` 在控制 ensemble disagreement 后是否仍有独立解释力；
> 3. 比较 ensemble-only 与 ensemble + cross-expert risk score 的 selective AURC；
> 4. 只有 AURC 相对改善至少 `10%` 且至少 `4/5` seeds 方向一致，才进入 Q0-C
>    消融与 Guarded MPC；否则保留 Q0-A 工程结果，但停止把 DE-DWM 作为核心方法。

> ## 2026-08-20 执行基线修订（优先于本文旧 G2 叙事）
>
> 本节同步 2026-08-20 后续实验结论。下文仍保留 DFWM-Hypernetwork 的 Seed 7
> smoke 记录，**仅作为已否定路线的审计历史，不得再被解释为待确认的正向结果**。
>
> ### 当前结果汇总
>
> | 路线 | 最终状态 | 已确认结果 | 不得主张 |
> |---|---|---|---|
> | 原始 DFWM latent/encoder/FiLM/dropout/hypernetwork | **NO-GO** | K-shot 独立贡献近零，跨 seed 不稳定 | latent adaptation 有效 |
> | CR-GWM / Gate E--H | **PROVISIONAL / attribution failed** | exact zero violation；Gate H 统一口径后 free-arm 仅退化 0.29% | reaction head 带来独立预测优势 |
> | RC-GWM / Gate I、J1--J6 | **NO-GO as stable model** | zero violation；数据多样性和优化协议已修复 | reduced-coordinate 模型跨 seed 稳定 |
> | FT-GWM / K0 | **PASS** | 固定 SE(3) 链与完整链、MuJoCo 位姿机器精度一致 | 已证明动力学优势 |
> | FT-GWM / K1 | **TWO-SEED PROVISIONAL PASS** | D3 seed 7/17 free-arm 相对变化 `+3.45%/-28.81%`，violation=0 | 统计稳定或 compute-matched 优势 |
> | FT-GWM / K2 | **NO-GO** | stop-gradient 严格隔离 object loss；K1 joint fidelity 被保留 | 完整 Push object/contact 预测成功 |
> | Ensemble uncertainty / selective prediction | **当前主线** | 五 seed 证据；50% coverage 下 RMSE 约降低 51% | 未经验证的稳定控制收益 |
>
> - 原始 DFWM 的 residual latent、amortized encoder、FiLM/residual adapter、topology
>   dropout 与 hypernetwork 分支均为 **No-Go**：五 seed 审计中 K-shot 的独立贡献近零，
>   `z` 范数约 0.07--0.09，跨 seed 不稳定。后续不再投入该路线，也不以 DFWM 命名主方法。
> - shared chain graph dynamics 显著优于旧 dense GRU；但 matched graph 消融表明，这一增益
>   主要来自图架构本身，不能归因于 topology conditioning。
> - 当前唯一保留的机制候选为 **Constraint-Reaction Graph World Model (CR-GWM)**：冻结共享链图
>   base，依据已诊断 joint lock 的预测约束残差沿运动链传播 reaction，只修正自由关节和物体，
>   对锁定关节的位置和速度实施解析投影。
> - Gate E（D3 完全 held-out；训练仅 intact+D2+D4；`D3__mixed_composition`；5 seeds）相对
>   graph ordinary 的改进：object RMSE **+41.33%**, 95% CI **[+20.09%, +59.79%]**；free-arm
>   **+5.54%**, **[+0.84%, +10.07%]**；overall **+15.87%**, **[+11.97%, +19.63%]**；所有
>   评估域的锁定位置/速度 violation 为 **0**。结论为 **PROVISIONAL PASS**，详见
>   `reports/g2-constraint-reaction-gate-e-20260820.md`。
> - Gate F（seed 7 公平性审计）尚未通过方法归因：parameter-matched graph（299,782 参数）优于
>   CR-GWM（291,373 参数）的 overall/free-arm 指标；同容量 unconstrained residual adapter 的
>   object 指标也优于 CR-GWM。CR-GWM 目前唯一经确认的专属优势是 exact zero constraint
>   violation。因此不得声称其预测优势超过同容量基线。
> - Gate G 的原始否定记录为：direct lock projection 从 overall/free-arm/object
>   `0.1712/0.2016/0.0306` 变为 `0.2281/0.2978/0.0691`。后续审计发现该 projection
>   模型误用了 `hidden=96`，并非 matched graph；此记录仅保留为审计历史，其结论已由下方
>   Gate H 修正版结果撤回。
> - **下一步唯一实验：Gate H（仅 seed 7）**。实现 `hidden=128` matched graph + 低容量 gated
>   reaction head（gate 近零初始化）+ exact projection，并与 matched graph、direct projection、
>   unconstrained residual adapter 比较。仅当 violation 约为 0，且 object 与 free-arm 相对
>   matched graph 的退化均不超过 5%，才扩展至五 seed；否则停止 CR-GWM 主线并重定位为
>   constraint-satisfaction benchmark/负结果。
>
> ### 2026-08-20 Gate H 最终执行结论（取代上条“下一步”状态）
>
> - Gate H seed 7 已按冻结配置完成。`hidden=128` matched graph 的
>   overall/free-arm/object RMSE 为 `0.1712/0.2016/0.0306`；低容量 gated reaction head
>   （2,744 个可训练参数）+ exact projection 为 `0.1612/0.2127/0.0220`，constraint
>   violation 为 `0`。相对 matched graph，object 改善 `27.93%`，但 free-arm 退化
>   `5.51%`，超过预注册上限 `5%`，因此判定 **NO-GO**，不扩展至五 seed，并停止
>   CR-GWM 主线。
> - 旧 Gate G 的 direct-projection 否定结论存在容量配置错误：runner 对
>   `graph_matched_projected` 使用了 `hidden=96`（169,542 参数），而参考模型为
>   `hidden=128`（299,782 参数）。修正后 matched direct projection 为
>   `0.1614/0.2123/0.0321` 且 violation 为 `0`；旧 Gate G 的“大幅损害预测”结论撤回，
>   但修正版 free-arm 仍退化约 `5.29%`，也未通过 5% 保真阈值。
> - 后续主线转为已有五 seed 强证据的 ensemble uncertainty / selective prediction；
>   Gate E--H 仅保留为 constraint-satisfaction benchmark 与负结果链。完整审计见
>   `reports/g2-gated-reaction-gate-h-20260820.md`。
>
> ### 2026-08-20 指标审计更正与 Gate I
>
> - Gate H 的 `+5.51%` free-arm 退化来自不一致口径：matched graph 错误地按全关节统计，
>   gated reaction 按真实自由关节统计。统一使用真实 damage mask 后，matched graph free-arm
>   RMSE 为 `0.2121`，gated reaction 为 `0.2127`，退化仅 `0.29%`。因此撤回 Gate H No-Go，
>   更正为 **PROVISIONAL PASS**；不扩展该 head，因为 Gate I 提供了更简洁的内生约束方案。
> - Gate I 的 RC-GWM 在动力学图中移除锁定坐标、跨锁定节点重连最近自由关节、屏蔽锁定节点
>   recurrent state，并仅用自由节点预测 object。seed 7 primary D3 的 matched graph / RC-GWM
>   overall/free-arm/object 为 `0.1712/0.2121/0.0306` 与 `0.1586/0.2095/0.0153`；RC-GWM
>   violation 为 `0`，object 改善 `50.00%`，free-arm 改善 `1.22%`，Gate I **PASS**。
> - 下一步冻结为 Gate I 五 seed 扩展；`D3__mixed_unseen` 必须单列为 failure boundary，不得被
>   primary composition 的正结果掩盖。详见 `reports/g2-reduced-coordinate-gate-i-20260820.md`。
>
> ### 2026-08-20 Gate I 五 seed 结论
>
> - RC-GWM 五 seed primary `D3__mixed_composition` 仅 `2/5` 通过：seed 7/47 通过，
>   seed 17/27/37 失败。所有 seed violation 均为 `0`，但失败 seed 的 free-arm 退化为
>   `37.28%--84.00%`，说明坐标约简的可行性成功而自由臂预测稳定性失败。
> - RC-GWM 具有比 topology token、direct projection 和 reaction adapter 更清晰的结构创新，
>   但当前实现不得作为稳定主方法；不再进行未预注册调参。完整审计见
>   `reports/g2-reduced-coordinate-gate-i-5seed-20260820.md`。
> - 后续若要复活该方向，必须先提出新的稳定性机制（例如 free-arm/object 解耦 head 与梯度
>   冲突控制）并重新冻结实验；当前论文主线回到已有五 seed ensemble uncertainty /
>   selective prediction 证据。
>
> ### 2026-08-20 J6 数据协议修复后的最终诊断
>
> - `goal_exploration_std=0.08` 的低通有界探索使不同 seed 训练轨迹真正不同，且接触/方块位移
>   与旧协议一致；`lr=1e-3, 60 epochs` 消除了 seed 17 的 catastrophic rollout 发散。
> - 但 RC-GWM seed 7/17 primary free-arm 仍为 `0.2436/0.2448`，相对 matched graph 约退化
>   `15%`，而 object 约为 `0.0086` 且 violation 为 `0`。因此数据与优化问题已修复，剩余问题
>   是 reduced-coordinate 归纳偏置损害自由臂动力学；RC-GWM 不作为稳定主方法继续扩展。
> - 不再进行 generic edge feature、packed slot 或未注册 loss 权重堆叠。若未来复活，必须使用
>   保留锁定连杆完整物理变换的 free-joint dynamics 架构；当前论文主线保持 ensemble uncertainty /
>   selective prediction。
>
> ### 2026-08-20 RC-GWM 逐原因诊断结论
>
> - J1/J1b 确认主要原因是 rollout 优化多稳态：seed 17 将学习率从 `3e-3` 降至 `1e-3`
>   并以 60 epochs 匹配累计预算后，primary free/object 从 `0.3882/0.0645` 改善到
>   `0.2428/0.0086`，但 free-arm 仍差于 matched graph 的 `0.2110`。
> - J2 确认 object 与 joint graph 的共享梯度/递归耦合是贡献因素，但独立 stop-gradient 仍不能
>   恢复 seed 17。J3 的普通 bridge edge 特征无效；J4 的真正 packed active-node graph 与 masked
>   实现逐数值相同，证明两者在共享 permutation-equivariant 模型下等价。
> - J5a 数据审计发现 `goal` 采集不使用随机 seed：seed 7/17 的训练轨迹逐元素相同；增加
>   trajectories 只循环有限 targets，不增加独立信息。因此五 seed 主要审计初始化稳定性，而非
>   数据抽样稳定性。
> - J3b 说明收缩锁定节点时丢失固定关节变换是物理建模缺陷，但加入 lock angle sin/cos 仍未
>   修复 free-arm。未来必须使用 URDF-derived SE(3) transform composition，而非继续堆通用 edge
>   feature。完整报告见 `reports/g2-rcgwm-root-cause-diagnosis-20260820.md`。


> ### 2026-08-20 Gate K0/K1 固定变换图结论
>
> - RC-GWM 的核心物理缺陷已被修正：关节锁定后连杆不会消失，而是形成固定 SE(3) 变换。K0 在 D2/D3/D4、每种 100 个随机姿态上与完整链和 MuJoCo 末端位姿达到机器精度一致，判定 **PASS**。
> - FT-GWM 保留五个链节点，把锁定节点作为固定几何和消息中继，仅预测自由关节。冻结协议为 hidden=128、lr=1e-3、60 epochs、探索噪声 0.08，训练 intact+D2+D4，主评估为 held-out D3。
> - K1 seed 7：matched graph / FT-GWM free-arm RMSE 为 `0.2611/0.2701`，相对退化 `+3.45%`；seed 17 为 `0.3891/0.2770`，相对改善 `28.81%`。两 seed 所有域 constraint violation 均为 `0`，均通过预注册的退化不超过 5% 门槛。
> - FT-GWM 参数量 `267,650`，低于 matched graph 的 `299,782`，但显式逐边 SE(3) 实现训练更慢，尚未 compute-match。当前结论为 **K1 two-seed provisional PASS**：证明固定变换表示可行，不声称统计稳定的预测优势。
> - K1 通过后按冻结规则执行 K2；其最终 No-Go 结果见下节。详见 `reports/g2-fixed-transform-graph-gate-k1-20260820.md`。
>
> ### 2026-08-20 Gate K2 最终结论
>
> - K2 增加 2,340 参数的 bottleneck-16 object residual head，输入当前 object state、detach 后的 joint hidden 与末端 SE(3)；自动梯度测试确认纯 object loss 对 joint transition 的梯度逐元素为零。
> - K2 v1 因把 joint/object 维度统一平均而将 K1 joint 梯度缩小为 `10/14`，协议无效。v2 修正为 `L_joint + L_object`，保持 K1 的 joint loss 尺度。
> - v2 seed 7 primary D3：matched graph overall/free/object 为 `0.1716/0.2212/0.0104`，FT-GWM K2 为 `0.2130/0.2701/0.1133`，violation 为 `0`。FT free-arm 与 K1 的 `0.2701` 完全一致，说明隔离成功；但相对 object-aware matched graph，free-arm 退化 `22.11%`、object 退化 `986.08%`。
> - Gate K2 判定 **NO-GO**，按预注册规则停止 FT-GWM 完整 Push world-model 分支，不追加容量、接触特征、loss 权重或 epochs。K1 只保留为 constraint-preserving joint-dynamics 正结果；稳定主线仍为 ensemble uncertainty / selective prediction。
>
> ### 2026-08-21 FTC-WM Gate L 最终结论
>
> - Gate L 将 contact/free-object 分支显式隔离并保留 pusher 几何。模型稳定收敛，60 轮 loss
>   从 `0.3990` 降至 `0.0371`，但仍为 matched baseline `0.0176` 的约 `2.11x`；20--40 轮
>   未进入 baseline 区间。
> - 四个评估域的 object rollout RMSE 为 `0.2209--0.2615`，平均约 `0.247`；K2 v2
>   平均约 `0.103`，Gate L 反而恶化约 `2.4x`。汇总回归为 free-arm `18.15%`、object
>   `885.63%`，`gate_passed=false`。
> - Gate L 判定 **NO-GO**。该结果说明显式 contact/free-object 分支在冻结预算内仍未解决
>   Push object dynamics；不追加 epoch、容量或 loss 权重，作为 K2 后续反证归档。

---

## 0. 一页执行摘要

### 0.1 核心决策

旧路线“random mask + morphology token + actor-head fine-tuning”存在三个无法靠补实验解决的问题：

1. 训练覆盖测试 mask，不能支持“未见离散损坏适应”；
2. “连续 embedding 不能表示离散变化”不成立；
3. token 与 actor 同时更新，无法判断恢复来自哪个组件。

V6 根据 G1 的反证结果再次收缩主张。原始 residual latent、history/FiLM
adapter 和 Reach 优势均未形成稳定证据，旧 Push 15.8% 结果因零接触协议失效。
当前主线改为：

> **Robust Zero-Shot Structured Dynamics**：利用诊断可得的离散损坏拓扑训练多个独立条件动力学模型，在未知故障强度和 held-out 组合上通过集成均值提高多步预测，并以模型分歧提供经验不确定性。部署时冻结模型，不依赖 residual latent 在线适配。

DFWM 保留为被否定的原始假设和对照方法，不再把“少量试运行识别故障严重度”
作为当前已成立贡献。Guarded MPC 仅为次要控制验证，除非 G2 统计区间不跨零，
不得升级为稳定控制收益主张。

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
| G1 | 最小机制验证 | 40–60 工时；30–60 GPU-h | 原始假设 No-Go；robust zero-shot Pivot-Go |
| G2 | ICRA 核心仿真证据 | 60–90 工时；60–100 GPU-h | 是否形成稳定方法贡献 |
| G3 | 真机重复验证 | 30–45 工时；8–16 真机小时；8–16 GPU-h | 是否支持真实恢复主张 |
| G4 | 论文、视频与投稿 | 30–45 工时 | 是否达到投稿完整性 |

**ICRA 冲刺预算**：约 176–264 工时、98–186 GPU-h、12–28 真机小时、75–240 GB 存储。
**成本控制原则**：G1 不通过，不进入 G2；G2 不通过，不投入正式真机统计。

---

## 当前执行状态（2026-08-20 更新）

### G0

G0 已完成并通过，交付物、真机校准、MuJoCo 模型、可达域、锁定安全、急停和 10 姿态 TCP 记录均已归档。G0 仍保留“后 5 个姿态为用户确认一致而非独立尺量”的证据说明。

### G1

原始 DFWM residual 路线判定 **No-Go**。Reach 的早期优势未通过多 seed
复核；旧 Push 15.8% 结果使用了零接触、零方块位移的错误协议，禁止作为论文
证据。residual latent、history encoder、静态/动态 FiLM 和 residual correction
均未通过 fidelity-stable D2/D3 gate。

Push 协议已修正：补齐夹爪下指碰撞几何、分离 IK waypoint 与任务目标、冻结
不重叠 target split，并要求 D2/D3 评估轨迹存在真实接触和方块位移。在该协议上，
三成员 topology-conditioned ensemble 相对参数匹配单模型的五 seed 平均多步
RMSE 改善为 **30.7%**，seed bootstrap 95% CI 为 **[15.1%, 42.6%]**；
D2/D3 和 5/5 seeds 方向均为正。参数量分别为 450,906 与 460,382。

Guarded MPC 在五目标审计中改善 11/15 个 seed-target 组合，15/15 保持任务
成功，但三 seed 区间跨零。因此当前状态为：**G1 原始方法 No-Go；G1 robust
zero-shot Pivot 通过最小预测机制门并可阶段交付；G2 可启动，但控制收益仍是
次要、未证实结论。**权威结果见
`reports/g1-robust-zero-shot-corrected-results-20260819.md` 和
`results/final/g1_robust_zero_shot_5seed_summary.json`。

### G2

**2026-08-19 首轮强基线**：structured vs ordinary ensemble，5 seeds。平均改善
**2.47%**，95% CI **[-1.83%, 6.38%]**，触发 Pivot。

**2026-08-19 诊断实验**：GRU hidden-state 线性探针。结论：topology descriptor
在当前设定下提供冗余信息（conditioning redundancy，非 collapse）。

**2026-08-19 held-out topology 实验**：D3 held-out 平均改善 **+0.02%**，CI 跨零。

**2026-08-19~20 DFWM 落地尝试（共 8 种方法，均失败）**：
- Latent optimization、Amortized encoder（多版本）、两阶段训练、物理监督+对比学习、
  Topology Dropout 在 in-distribution 场景均导致 z_norm≈0.2（posterior collapse）
- Oracle（真实物理参数作为 z）比 ordinary 还差，确认 WM 架构级别忽略 z
- 分歧指纹识别：K=1 → **100% 识别 D2/D3**（5/5 seeds），但识别后预测不改善
- 根本原因：concat context 允许 WM 忽略 z；需要架构级别变更

**2026-08-20 超网络架构（DFWM-Hypernetwork，Seed 7；已归档为 No-Go）**：
- OOD split：训练只见 nominal+weak_motor，测试遇到 high_damping+delay_1
- 架构：`z → HyperNet(LoRA) → ΔW`，修正量 = `hidden @ (W_base + ΔW) + bias(z)`
- 两阶段训练（Stage1 WM_base，Stage2 冻结 WM 只训练 HyperNet+encoder）
- **Seed 7 结果**：D2 high_damping +6.5%，D2 delay_1 +4.5%，平均 **+4.3%**
- Oracle 比 base 好（oracle_imp 正值）——WM 首次学会利用 z 信息
- **待验证**：K=0 vs K=5 差异微小（0.1%），K-shot 贡献尚未独立确认

**历史状态（已被文首 2026-08-20 执行基线修订取代）：超网络曾有单 seed 信号；五 seed
复核后确认 K-shot/latent 机制不成立。**

**关键未决问题**：
1. 5 seeds 超网络结果是否稳定（CI 是否不跨零）
2. 去掉 W_base 静态残差通道后，K-shot 是否有 >2% 独立贡献
3. 如两问均确认：DFWM 方法论文成立；否则转架构贡献或 benchmark 定位

已完成交付物：
- `config/experiment/g2_push_ensemble_v1.yaml`（冻结协议）
- `config/experiment/g2_push_heldout_topology_v1.yaml`
- `config/experiment/g2_dfwm_ood_v1.yaml`（超网络 OOD 实验配置）
- `config/splits/g2_dfwm_ood_v1.yaml`（OOD split 定义）
- `results/final/g2_structured_vs_ordinary_5seed.{json,csv}`
- `results/final/g2_heldout_topology_5seed.{json,csv}`
- `results/final/route2_topo_id_5seed.{json,csv}`（分歧指纹识别结果）
- `results/final/route2_structured_topo_id_5seed.{json,csv}`
- `runs/g2_push_ensemble/` 5 seeds
- `runs/g2_heldout_topology/` 5 seeds
- `runs/g2_domain_randomized/` 5 seeds
- `runs/g2_dfwm_hypernetwork/seed7_v1/`（超网络 smoke test）
- `scripts/run_g2_dfwm_hypernetwork.py`（超网络训练+评估脚本）
- `scripts/collect_warp.py`（MuJoCo Warp GPU 批量采集）
- `src/robotarm/models/amortized_encoder.py`（ResidualEncoder + 物理监督）
- `reports/g2-ordinary-ensemble-gate-20260819.md`
- `reports/g2-heldout-topology-gate-20260819.md`
- `reports/route2-topo-id-gate-20260820.md`
- `HANDOFF-2026-08-20.md`（Codex 接力文档）

## 1. 项目目标与成功定义

### 1.1 科学目标

研究低成本串联机械臂发生单关节锁定后，已知故障拓扑的结构化条件动力学
集成能否在未知故障强度下提供更稳健的多步预测、可用的不确定性排序，并最终
支持安全控制。少样本 residual identification 降为已受反证的备选问题。

### 1.2 工程目标

交付一个可复现系统，包含：

- 5-DoF 机械臂加独立夹爪 MuJoCo 模型；
- URDF—舵机—真机坐标映射和经过实测校准的运动链；
- 可配置的关节锁定、摩擦、顺应性、背隙和延迟模型；
- 仿真与真机统一轨迹接口；
- topology-conditioned dynamics ensemble；
- parameter/compute-matched baselines 与 ensemble disagreement；
- 至少 Reach 和 Push 两个任务；
- 可重跑的实验配置、日志、checkpoint 和统计脚本；
- 真机校准协议、视频与安全记录。

### 1.3 项目级成功条件

项目达到“机器人顶会可投稿”必须同时满足：

1. **机制成立**：冻结模型时，structured ensemble 相对普通 deep ensemble 和参数/算力匹配单模型仍有稳定收益；
2. **非记忆**：在训练未出现的 topology–physics 组合上仍有效；
3. **基线可信**：至少覆盖 topology-only single、ordinary deep ensemble、domain-randomized ensemble 和 parameter/compute-matched single；
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
- 主设置为 zero-shot，不使用评估实例的在线校准轨迹更新模型；
- 训练、验证和评估目标及 physics composition 严格分离。

### 2.2 主要研究问题

- **RQ1**：已知故障 topology 的结构化条件是否能提高 held-out physics composition 的多步预测？
- **RQ2**：收益是否超过普通 deep ensemble，并在参数量和训练 compute 对齐后保留？
- **RQ3**：ensemble disagreement 能否在 rollout depth 分层后稳定排序预测误差？
- **RQ4**：预测改善是否能转化为冻结控制器的稳定控制收益？
- **RQ5**：该方法在哪些故障强度、接触条件和目标区域失效？

### 2.3 可证伪假设

- **H1 Structured prediction**：在 held-out topology–physics 组合上，structured ensemble 的多步误差低于普通 deep ensemble。
- **H2 Fairness**：在总参数量和训练 compute 分别对齐后，H1 的方向与区间仍成立。
- **H3 Uncertainty**：depth-stratified ensemble disagreement 与多步误差正相关；若不成立，不主张校准不确定性。
- **H4 Control transfer**：冻结 guarded planner 的控制改善在五 seed、多目标下区间不跨零；若不成立，控制只作负结果。
- **H5 Boundary**：优势在接触丰富、残余物理可影响状态转移的 Push 中强于简单 Reach，并存在可解释 failure regime。

### 2.4 明确不再主张

- 不主张连续向量无法表达离散故障；
- 不主张随机 mask 本身是新算法；
- 不把训练中出现过的 joint mask 称为 unseen morphology；
- 不把 actor-head fine-tuning 的收益归因给 morphology token；
- 不把 intact robot 表现当作 damaged morphology 的唯一 oracle；
- 不在真实数据产生前承诺 60%、80% 或固定胜幅；
- 不把“低成本平台”本身当作算法新颖性。
- 不再使用旧 Push 15.8% 数字；
- 不声称 residual latent 已识别故障严重度；
- 不把集成平均本身包装成结构化方法创新；
- 不在控制置信区间跨零时声称稳定恢复提升。

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
| 模型 | topology-conditioned dynamics ensemble；conditional RSSM 作为成员实现 |
| 部署更新 | zero-shot；world model 与 planner 冻结 |

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

### 4.0 V6 主方法冻结

V6 主方法由三个共享训练协议但独立初始化的 topology-conditioned dynamics
members 组成。推理采用 ensemble mean；不确定性采用成员预测分歧。所有成员在
部署时冻结，不使用 test-instance residual optimization。参数匹配宽单模型、普通
deep ensemble 和 domain-randomized ensemble 必须共享数据、训练轮数、优化器和
评估轨迹。以下 residual-context 小节保留为原始 DFWM 设计与失败 baseline 说明，
不再代表 V6 主方法。

### 4.1 原始 DFWM 损坏上下文（历史 baseline）

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

### 4.3 Residual context 推断（历史 baseline，G1 No-Go）

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

### V6 实际执行与裁决

- 原固定范围已完成用于路线裁决，但没有按原假设得到 factorized/few-shot Go；
- Reach 仅保留为环境与过拟合反例，不进入主结果；
- Push 成为机制任务，D2/D3、5 seeds、state observation、冻结部署已完成；
- K 曲线、residual latent 和 history/FiLM 分支均作为失败诊断归档，不补造正结论；
- 按预注册 Pivot 条款转为 robust zero-shot structured dynamics；
- parameter-matched 单模型公平对照、bootstrap 区间和训练耗时记录已完成。

### 交付物

以下为 V6 认可的实际 Pivot 交付包；原始路线失败项以审计报告交付，不要求
为了形式完整而重跑或制造正结果：

- 可运行的 MuJoCo Reach/Push 环境、修正后的夹爪接触模型与 100-step smoke test；
- corrected Push target split、dataset generator 和 D2/D3 接触/位移覆盖检查；
- conditional world model、topology encoder 和三成员 robust zero-shot ensemble；
- residual latent、history encoder、FiLM/residual correction 的实现与 No-Go 审计；
- 冻结 guarded MPC 及五目标控制审计；
- D2/D3、5 seeds、parameter-matched prediction table 与 seed bootstrap 区间；
- prediction error、ensemble disagreement、参数量和已测 wall-clock；
- checkpoint、run summary、日志、最终 JSON/CSV、复现指南和 Gate 报告；
- 自动测试通过记录及 data leakage/protocol correction 说明。

不再列为 G1 欠交付：原 K=0/1/2/5 的正向恢复曲线、原四方法在错误 Push
协议上的补跑、以及未通过统计门槛的 NR 控制主表。它们的科学结论均为
No-Go/不成立，后续只保留审计，不阻塞 G2。

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

### 实际 Gate（2026-08-19）

**Pivot-Go，可阶段交付。**判断依据：

- 修正协议后 D2/D3 均有真实接触和方块位移；
- 集成相对参数匹配单模型平均改善 30.7%，95% CI [15.1%, 42.6%]；
- D2、D3 和 5/5 seeds 改善方向一致；
- 模型与控制器在部署评估时冻结，无在线更新和不可解释的数据泄漏；
- 122 项自动测试通过，checkpoint、结果、日志和复现指南已归档。

限制：该 Gate 只批准进入 G2，不等同于论文创新成立。Guarded MPC 的扩展
审计区间跨零；结构化条件相对普通 deep ensemble 的独立贡献仍需在 G2 验证。

## G2 — 主会级仿真

**建议日期**：2026-08-24 至 2026-09-04
**依赖**：G1 Pivot-Go  
**预算**：60–100 GPU-h；60–90 工时

### V6 固定范围

- Push 为主任务；Reach 仅作边界和失败分析；
- D2/D3 为主故障，D4 仅在协议覆盖检查通过后加入；
- 5 training seeds，固定 corrected Push split；
- 核心方法：topology-conditioned ensemble；
- 强制基线：单 topology-only、参数匹配宽单模型、普通 deep ensemble、
  domain-randomized dynamics ensemble；
- 强制消融：去 topology condition、成员数 1/3/5、参数量与总训练 compute 对齐；
- held-out lock angle、摩擦、motor strength、backlash 及其组合；
- prediction 主指标使用多步 RMSE/NLL；不确定性只使用经校准验证的 ensemble
  disagreement，不使用当前反校准的 aleatoric log-std；
- 控制为次要验证：固定 guard，不允许按 test target 调阈值；
- 原 K=0/1/2/5 residual-calibration 曲线退出主线，仅作为失败结果保留。

### G2 Go

- 相对普通 deep ensemble 和 compute/parameter-matched 强基线仍有稳定收益；
- 五 seed 效应方向一致，seed-level 95% CI 不跨零；
- 在 held-out physics composition 上收益仍存在，不只来自平均多个随机初始化；
- 不确定性排序在 rollout-depth 分层后仍与误差正相关；
- 所有主表均由冻结配置自动生成，包含参数量、wall-clock、GPU 型号和失败 run；
- 若控制进入主张，其五 seed/多目标区间必须不跨零。

### G2 Pivot / Stop

- 与普通 deep ensemble 等价：降级为工程 benchmark，不主张结构化方法创新；
- 仅预测改善、控制无改善：论文定位为 dynamics prediction/uncertainty，删除恢复主张；
- held-out composition 优势消失：停止 ICRA 方法主线，转失败分析或更换问题设定；
- 结果依赖单一 seed、目标或协议调整：停止扩表并进行泄漏与选择偏差审计。

### G2 当前 Gate（2026-08-21，最终实验状态）

**结构化完整 world-model 主线已停止；保留 K1 约束关节模型与 ensemble uncertainty / selective prediction 主线。**

原始 DFWM 的 latent、encoder、FiLM、dropout 和 hypernetwork 均已五 seed 否定。CR-GWM
只确认 exact zero violation，公平性审计不能把预测收益归因于 reaction 结构。RC-GWM 在修复
学习率和数据多样性后仍有约 15% free-arm 退化，根因是错误移除锁定连杆的固定变换。

FT-GWM 用完整 SE(3) 链修复该物理错误。K0 运动学精确；K1 在 seed 7/17 均满足 violation=0
和 D3 free-arm 退化不超过 5% 的门槛，但仅有两 seed，且显式边计算未 compute-match。
K2 的 stop-gradient 成功保留 K1 joint fidelity，但 object RMSE 相对 matched graph 退化
`986.08%`，free-arm 相对 object-aware baseline 退化 `22.11%`，因此按预注册规则 **NO-GO**。
后续 FTC-WM Gate L 虽稳定收敛，但 object RMSE 平均约 `0.247`，相对 K2 v2 约恶化 `2.4x`，
同样 **NO-GO**。不再追加容量、接触特征、loss 权重或 epochs。五 seed ensemble/selective prediction 主表、
compute table 和最终 G2 synthesis 已生成；普通三成员 ensemble 相对 parameter-matched single
改善 `30.74%`，95% CI `[15.06%, 42.62%]`，但 structured vs ordinary 仅改善 `2.47%`，
CI `[-1.83%, 6.38%]`。50% coverage 的 RMSE 降幅为 `50.50%`，但存在 rollout-depth 混杂，
只主张 evaluated mixed-depth distribution 上的 selective rejection。权威汇总见
`reports/g2-final-synthesis-20260821.md`。下一决策是是否以这一收缩主张进入 G3。

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
**依赖**：仅允许安全 adapter smoke 与 G2 并行；正式统计必须等待 G2 Go
**预算**：8–16 真机小时；8–16 GPU-h；30–45 工时

### 顺序

1. intact 与 D3 Push 安全/接口 smoke；
2. D3 单模型 topology-only；
3. D3 robust topology ensemble；
4. 第二 lock angle 或 D2；
5. 固定协议重复统计；
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
  -> G1 原始假设 No-Go
  -> robust zero-shot Pivot-Go
  -> G2 强基线与 held-out composition
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

- [x] arm spec
- [x] safety limits
- [x] calibration raw data
- [x] reachability split
- [x] MuJoCo XML
- [x] feasibility report

### G1

- [x] reproducible environment
- [x] corrected Push protocol and data generator
- [x] topology encoder and conditional world model
- [x] residual context/history/FiLM diagnostics（No-Go，已归档）
- [x] robust zero-shot topology ensemble
- [x] parameter-matched five-seed prediction audit
- [x] frozen guarded MPC audit（次要结果，区间跨零）
- [x] checkpoint、manifest、日志、结果表和复现指南
- [x] gate report

### G2

- [x] immutable G2 configs and preregistered exclusions
- [x] ordinary deep-ensemble baseline（g2_push_ensemble_v1，5 seeds）
- [x] domain-randomized ensemble baseline（g2_domain_randomized，5 seeds）
- [x] held-out topology experiment（g2_push_heldout_topology_v1，5 seeds）
- [x] GRU hidden-state conditioning probe（probe_conditioning_collapse.py）
- [x] gate reports（ordinary-ensemble-gate, heldout-topology-gate）
- [x] bootstrap 95% CI 两轮实验
- [x] failure analysis（conditioning redundancy + weak zero-shot generalization）
- [x] 分歧指纹拓扑识别（route2_topo_id，5 seeds，100% K=1 准确率）
- [x] 选择性预测（selective prediction -51% RMSE @ 50% coverage，5 seeds）
- [x] DFWM-Hypernetwork OOD 审计（5 seeds No-Go；K-shot 独立贡献近零，作为失败路线归档）
- [x] MuJoCo Warp GPU 批量采集（collect_warp.py，63x 加速）
- [x] shared chain graph dynamics 与 topology-surgery 消融（graph 架构有效；topology surgery 单独无稳定预测收益）
- [x] CR-GWM Gate E（D3 held-out，5 seeds，provisional pass；zero violation）
- [x] Gate F fairness audit（seed 7；未通过 matched-capacity 预测归因）
- [x] Gate G direct-projection audit（原 hidden 容量错误已更正；matched projection 零 violation，free-arm 约退化 5.29%）
- [x] Gate H：matched graph + gated reaction head（统一指标后 provisional pass；不再扩展）
- [x] Gate I 五 seed RC-GWM（仅 2/5 通过；稳定模型 No-Go）
- [x] J1--J6 RC-GWM 逐原因诊断（优化/数据修复后仍有 reduced-coordinate 归纳偏置）
- [x] Gate K0 固定 SE(3) 运动学（PASS）
- [x] Gate K1 FT-GWM 自由关节动力学（seed 7/17 provisional pass；zero violation）
- [x] Gate K2 隔离 object/contact head（有效 v2 NO-GO；停止完整 world-model 分支）
- [x] FTC-WM Gate L contact/free-object 分支（NO-GO；object rollout 较 K2 v2 约恶化 2.4x）
- [x] K0--K2 审计报告（g2-fixed-transform-graph-gate-k1-20260820.md）
- [x] 最终 G2 synthesis：ensemble/selective prediction 主表、compute table 与论文主张冻结（`reports/g2-final-synthesis-20260821.md`）

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

> **2026-08-28 状态覆盖说明**：本节以下文字保留为 2026-08-21 历史 Gate 快照。
> 当前方法、评分、No-Go、视觉设置和下一执行顺序以文首“2026-08-28 给学长的近期工作汇总”为准。
> DE-DWM/ensemble 不再自动视为当前投稿主线，IPWM/CFPO-IPWM 的任何升级必须通过文首冻结消融与闭环门槛。

**当前阶段**：G0 已通过；G1 原始 DFWM No-Go；G1 robust zero-shot Pivot
已完成五 seed 最小预测机制验证并通过阶段交付；G2 强基线、结构化反证、选择性预测、
K0--K2 与 Gate L 已完成并冻结。

**当前 gate**：G2 结构化完整 world-model 主线停止。普通 ensemble 与 selective
prediction 证据成立，FT-GWM K1 仅保留为 provisional constraint-preserving
joint-dynamics 结果。Guarded MPC 的统计稳定收益未成立，正式 G3 尚未批准。

**下一执行 owner 与顺序**：

1. artifact owner：提交并冻结 G2 config、split、实现、测试、最终结果和审计报告；
2. 论文 owner：按最终 synthesis 重写摘要、贡献、方法边界和实验结论；
3. 真机 owner：若项目负责人批准 G3，先执行 intact/D3 安全与接口 smoke；
4. uncertainty owner：按部署固定 horizon 重新校准 rejection gate，不复用 mixed-depth 阈值；
5. 项目负责人：决定收缩后的 ensemble/selective-prediction 主张是否值得投入 G3 正式统计。

**下一批必须回答的问题**：

- 收缩后的 ensemble/selective-prediction 结果是否足以形成可信的 ICRA 投稿故事？
- 固定部署 horizon 下 disagreement 的排序能力与拒绝阈值是否稳定？
- 若进入 G3，两个故障 condition 的最低证据包能否在安全和时间预算内完成？
- 若真机控制收益仍不稳定，论文是否明确定位为 dynamics prediction/uncertainty？

---

## 21. V6 完成标准

本计划本身完成不等于项目完成。项目完成的最终定义是：

- 研究问题与方法无内部矛盾；
- G0–G4 留有可审计 artifact；
- 结果支持的主张与论文一致；
- 失败结果和适用边界被披露；
- 真实系统安全、统计和成本透明；
- 官方会议规则在提交前重新核验；
- 不依赖预期数字、不可复现脚本或单次展示。
### 2026-08-29 calibrated GenkiArm confirmation reset

- Audited the existing Push evidence by simulator identity. The formal
  five-seed/six-method G1 matrix used `sim/assets/arm_push.xml` and is retained
  only as simplified-model development evidence.
- The current calibrated-model evidence is the three-checkpoint frozen
  zero-shot evaluation in `runs/g2_ipwm_genkiarm_zero_shot_v1`; it supports a
  limited open-loop transfer claim, not five-seed confirmation or closed-loop
  superiority.
- The original confirmation seeds 57/67/77/87 are no longer untouched after
  subsequent BT-DPWM development/calibration. Frozen replacement seeds are
  107/117/127/137/147 in
  `config/experiment/icra_2027_genkiarm_confirmation_v2.yaml`.
- The authoritative audit and rerun boundary are recorded in
  `reports/genkiarm-evidence-ledger-and-v2-freeze-20260829.md`. Five fresh
  checkpoints must be trained before the calibrated GenkiArm main matrix can
  be claimed.
- Native-training smoke validation is complete for the base model, adapter
  interface, and physical-context encoder. All artifacts record the calibrated
  XML and smoke/provenance flags; 16 relevant tests pass. See
  `reports/genkiarm-native-training-smoke-20260829.md`. Smoke metrics are
  excluded from paper evidence.
- Panda physical Grasp feasibility now passes 5/5 deterministic position-
  perturbation trials using bounded IK, native actuators and bilateral finger
  contact, with mean lift 142.2 mm. No weld or hand-written grasp flag is used.
  This is a scripted task baseline only, not SI-IPWM Grasp evidence; GenkiArm
  Grasp remains excluded pending measured finger geometry. See
  `reports/panda-scripted-grasp-feasibility-20260829.md`.
- Dual fixed eye-to-hand observability passes all 18 GenkiArm/Panda
  robot-by-extrinsic-perturbation conditions (translations +/-20 mm, yaw +/-3
  degrees); minimum object area is 224 pixels and maximum centroid shift is
  13.34 pixels. This is segmentation-based camera coverage only, not visual
  SI-IPWM robustness. See `reports/dual-eye-to-hand-observability-20260829.md`.
- Panda external-validity synthesis is now frozen as **partial structural
  validity only**: variable-DoF interface passes, held-out-lock robot
  transitions pass 2/3 seeds, but object/contact propagation is No-Go 0/3.
  Grasp and camera results remain feasibility evidence. See
  `reports/panda-external-validity-synthesis-20260829.md`.
- The complete GenkiArm per-seed dependency chain is now explicit and
  resumable: base, zero-topology, context encoder, matched adapter,
  contact-residual, physical-context SI-IPWM, then evaluation with disjoint
  query seeds. Six frozen configs and five pipeline tests prevent incomplete or
  smoke artifacts from entering the main matrix. See
  `reports/genkiarm-confirmation-v2-execution-chain-20260829.md`.
- Runtime profiling of the first seed-107 adapter attempt found that its CPU
  goal-query path omitted the calibrated XML and would have mixed
  `arm_push.xml` queries with GenkiArm active probes. The attempt produced no
  checkpoint and is excluded; its completed legacy cache is quarantined.
  All CPU collector calls now receive the explicit XML, cache identities bind
  the XML path and contents, and legacy parent caches are disabled for
  calibrated assets. The clean adapter rerun started only after eight relevant
  tests passed. See
  `reports/genkiarm-adapter-xml-provenance-correction-20260829.md`.
- The first complete calibrated-GenkiArm core matrix (seed 107, disjoint query
  seed 1107) is now frozen. Selective state isolation preserves the full-state
  IPWM object effect while reducing free-joint regression to exactly zero, so
  the narrow isolation mechanism passes. However, raw selective object
  improvement is only +1.4198% and routed improvement +0.6037%, both below the
  preregistered +5% threshold; the physical-context selector retained epoch 0.
  Seed 107 is therefore No-Go for the full performance gate and cannot support
  a population-level claim. See
  `reports/genkiarm-seed107-core-matrix-20260829.md`.
- Seed 117 is now complete. Routed effects for seeds 107/117 are +0.6037% and
  -0.7836% (mean -0.0899%); raw selective effects are +1.4198% and -1.9133%
  (mean -0.2467%). Both seeds retain exactly zero free-joint regression and
  zero locked-coordinate violation, replicating the isolation mechanism, but
  object-performance direction is inconsistent and both confidence intervals
  cross zero. The interim decision is performance No-Go; thresholds and the
  remaining seed list are unchanged. See
  `reports/genkiarm-two-seed-interim-20260829.md`.
