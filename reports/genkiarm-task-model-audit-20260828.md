# GenkiArm Push/Grasp 模型可用性审计（2026-08-28）

## 裁决

现有 `arm_push.xml` 是与实物标定尺寸相符的简化碰撞链，但没有原始 CAD 外观，且正式实验
脚本均硬编码该文件。历史结果因此只能证明简化仿真环境中的机制表现，不能表述为“在给定
实际机械臂模型上的 Push/Grasp 验证”。

本次新增 `sim/assets/genkiarm_push.xml` 作为 GenkiArm Push 的主模型入口。它同时包含：

- 来自 `hardware/arm_spec.yaml` / `genkiarm_calibrated.urdf` 的五关节轴、限位、连杆长度、
  J5--tool 与 TCP 偏置；
- 用户提供的 AA--GG CAD 网格，仅用于外观渲染；
- 与标定 URDF 对应的 primitive collision proxies，用于稳定、可解释的接触动力学；
- 一个 40 mm 方块、桌面和两个固定 eye-to-hand 摄像头；
- 与现有 14 维 Push 状态和五维控制接口兼容的 MuJoCo 合同。

该模型成功通过 MuJoCo 加载、CAD/碰撞隔离、双外部相机和环境接口测试。数值逆运动学检查
能以小于 `2e-8 m` 的位置误差到达方块中心。冻结的 400-step approach--push smoke
产生 332 个 tool/pusher--block 接触步，并使方块位移 `0.13435 m`；自动回归门槛保守冻结为
至少 20 个接触步和 `0.02 m` 位移。这证明工作空间和接触链未因替换为标定模型而失效，
但不是学习方法的性能结果，也不计入论文主表。

## 仍未完成，禁止过度声明

1. 质量、惯量、关节阻尼、摩擦、执行器强度和接触参数仍是保守占位值，不是辨识结果。
2. `GG.stl` 是单一整体网格，包围盒约为 `45.85 x 21.47 x 86.92 mm`；它没有独立左右
   指爪、开合关节或可验证的指尖碰撞面。
3. `genkiarm_calibrated.urdf` 明确把 gripper opening 作为独立 servo，但当前仅建模五个臂
   关节和固定 tool；`hardware/joint_map.yaml` 虽给出 servo 6 映射，也没有指爪几何与行程。
4. 因此当前模型可进入 Push smoke/机制验证，但在完成接触参数随机化和任务可达性回归前
   不能进入正式主表；当前任何 Pick/Grasp 数字都不能作为物理抓取证据。

## Grasp 主表准入条件

必须从实物/CAD补齐并版本冻结：左右指爪几何、最大/最小开口、开合机构、servo-6 到开口
宽度的映射、指尖材料/摩擦，以及方块尺寸/质量。仿真还必须证明：

- 不允许理想 weld 或根据距离直接写入 `grasped=True`；
- 抓取成功由双指接触、闭合和持续抬升共同判定；
- intact 与 held-out lock 均使用相同控制预算和成功判据；
- 五 seeds、多目标/姿态/摩擦/质量组合以及失败边界均有独立 evaluation episodes。

## 论文用语

在动力学辨识完成前，准确用语是：

> calibrated-kinematic GenkiArm model with CAD visuals, collision proxies,
> and randomized provisional dynamics

不得使用 `fully calibrated dynamics`、`digital twin` 或把简化模型称为实际机械臂。
