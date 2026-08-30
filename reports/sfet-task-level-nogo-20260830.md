# SFET 机器人任务级快速验证：No-Go记录

时间：2026-08-30 20:12（Asia/Shanghai）

## 决策

**No-Go / insufficient valid evidence。** 截至本记录，不能声称 SFET 在原始 5-DoF 机械臂的 three-trial Push 恢复中具有巨大优势。

20:22追加：必要载体矩阵已经跑完。单步解析运输在D2成功率与hard-mask完全相同，在D3三种子均无法接触。此前命名为oracle IK的方法实际不使用隐藏真值，应改称可部署的fault-aware constrained IK强基线；它几乎解决了当前任务。因此No-Go不仅来自模型真实性缺口，也来自任务定义被简单解析基线覆盖。

## 为什么停止

任务要求使用物理原始机械臂的精确模型。现有 `sim/assets/arm_push.xml` 的运动学链与物理原始臂对应，但碰撞几何简化，惯量、摩擦、执行器和接触参数未通过真机系统辨识。因此继续扩大种子只会增加代理模型上的工作量，不能补足所要求的真实性证据。

## 完整必要载体开发数字

- D2 hard-mask成功率（7/17/27）：20/0/20%；analytic transport：20/0/20%；constrained IK：100/100/100%。
- D3 hard-mask成功率：0/0/0%；analytic transport：0/0/0%；constrained IK：80/100/100%。
- D2 analytic transport终点误差只在seed7改善约13.3%，seed17和27变差。
- D3 analytic transport与hard-mask终点误差完全相同，接触率均为0%；constrained IK接触率全部100%。
- 所有方法最大锁定动作违反均为0。

结果文件：`results/diagnostics/sfet-original-arm-carrier-screen-20260830.json`。这是运动学任务模型上的开发消融，不是真机证据；它足以否决单步解析载体，却不足以验证three-trial SFET。

## 缺失项

- D2/D3 × 3 seeds 完整矩阵；
- 每个故障严格三条完整校准轨迹；
- SFET Broyden 适配与冻结 IPWM 响应；
- 同预算 ridge/BC、HCAR 与原 IPWM；
- 未见目标、动作 regret、任务效果误差；
- 硬约束以外的关键消融；
- 精确动力学模型或真机目标环境。

## 最小恢复路径

1. 后天真机先采集 intact、D2、D3 各三条预注册安全校准 Push，记录时间戳同步的关节位置/速度、控制命令、双 eye-to-hand 方块 XY 和接触事件。
2. 用 intact 日志校正执行器尺度、阻尼和时延；用接触日志校正桌面/方块摩擦和接触参数。标定集与评价目标隔离。
3. 冻结模型后，在未见目标上比较 hard-mask、ridge/BC、HCAR、原 IPWM 和 SFET；全部重复及失败保留。
4. 仍使用门槛：成功率至少 +20pp 或误差/regret 至少下降 50%，3/3 seeds 方向一致并通过消融。

## 可向学长陈述的一句话

“我们完成了SFET实现和必要载体消融，但单步运输在D2/D3均未提高成功率；进一步审计发现此前标成oracle的方法其实是可部署约束IK，并已取得80--100%成功率，说明当前简单Push任务主要是运动学问题，无法支撑世界模型优势。后续若继续，必须共享约束IK后研究未知接触动力学，而不能继续与弱hard-mask比较。”
