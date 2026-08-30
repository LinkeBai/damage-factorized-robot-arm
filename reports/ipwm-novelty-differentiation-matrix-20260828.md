# IPWM差异化查新矩阵（冻结草案）

**日期**：2026-08-28  
**用途**：回应“idea可能已有近邻，必须做出差异化”的学长意见。  
**状态**：文献初筛完成；正式投稿前仍需按关键词、引文链和目标会议继续系统核验。  

## 核心问题

本文要解决的不是一般故障检测，也不是一般世界模型动作一致性，而是：预训练世界模型面对训练未见的机械臂关节锁定时，如何用少量安全数据保持硬约束，恢复故障对自由关节、接触和物体的影响传播，并提供可用于闭环规划的反事实动作预测。

## 近邻矩阵

| 近邻方向 | 代表工作 | 已覆盖内容 | 不能作为本文创新 | 本文必须证明的差异 |
|---|---|---|---|---|
| 动作可控世界模型 | WorldSync；CoCo | 同状态多动作、action-following、intervention-effect alignment | “加入动作反事实/排序损失” | 故障约束改变后的动作响应传播，并在未见锁定上改善闭环 |
| 干预与模块化因果动力学 | Variational Causal Dynamics；WM3C | 稀疏机制变化、因果组件、组合泛化与部分可识别性 | “因果干预”“可组合组件” | 已知关节锁定约束与少样本故障响应的机器人专属可验证分解 |
| 结构化/混合机器人动力学 | Differentiable Newton--Euler；NeRD；Nimble | 解析动力学、物理参数学习、接触约束和可微求解 | “解析物理+神经残差”“学习约束力” | 低维故障响应是否足以迁移到未见锁定并支持世界模型规划 |
| 零样本动力学迁移 | Augmented World Models；system-invariant dynamics | 物理参数变化、上下文适配和零样本域泛化 | “变化动力学下适配世界模型” | 离散自由度丢失、严格锁定约束和沿运动链/接触传播 |
| 关节故障容错 | fault-tolerant RL / locked-joint locomotion | 多故障训练、隐式故障估计、策略恢复 | “关节锁定后仍能控制” | 少样本世界模型适配、反事实预测归因和跨机械臂/跨任务泛化 |
| 图故障传播 | 机械臂故障诊断GNN | 沿关节拓扑传播诊断特征 | “沿运动链传播故障” | 传播变量必须预测状态和接触后果，而非仅分类故障 |

## 当前允许的差异化主张

在完成正式实验前，只允许提出以下研究假设，不允许宣称已经成立：

> 将未见关节锁定分解为解析硬约束和低维任务相关故障响应，并让该响应沿机械链与接触关系进入反事实世界模型，可能比全局状态残差更数据高效、更可迁移，并更适合闭环动作比较。

## 强创新成立条件

1. 相同数据、容量和预算下，完整方法稳定优于仅投影、全局残差、图掩码残差和原IPWM；
2. 学到的响应变量能解释free-joint与object反事实差分，而不是只复现MuJoCo内部标签；
3. 训练部分锁定关节后能泛化到未见关节和锁定角度；
4. 在第二种机械臂上无需重新定义机制，仅替换结构描述即可适配；
5. Push与Grasp均改善动作排序和闭环任务结果；
6. 预测改善、排序改善和控制改善形成配对统计证据链；
7. 去除路径结构或改为同容量无结构模型后，核心收益显著下降。

若上述条件不成立，论文应定位为约束安全的工程组合或负结果，不能宣称强创新。

## 初筛原始来源

- WorldSync: <https://arxiv.org/abs/2608.24885>
- CoCo / Action Response Consistency: <https://arxiv.org/abs/2608.04653>
- Variational Causal Dynamics: <https://arxiv.org/abs/2206.11131>
- WM3C: <https://openreview.net/pdf?id=XMgpnZ2ET7>
- Augmented World Models: <https://proceedings.mlr.press/v139/ball21a.html>
- Differentiable Newton--Euler: <https://proceedings.mlr.press/v120/sutanto20a.html>
- Nimble differentiable rigid-body/contact dynamics: <https://www.roboticsproceedings.org/rss17/p034.pdf>
- NeRD: <https://neural-robot-dynamics.github.io/>

## 下一轮查新关键词

`locked joint`, `jammed joint`, `actuator failure`, `constraint reaction`, `Lagrange multiplier identification`, `fault-tolerant model predictive control`, `few-shot dynamics adaptation`, `counterfactual robot world model`, `cross-embodiment dynamics generalization`。
