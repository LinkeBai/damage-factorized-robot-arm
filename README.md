# IPWM Robot Arm

面向低成本五自由度机械臂加独立夹爪的关节锁定恢复仿真、标定与实验仓库。当前以
原始 5-DoF 机械臂为主平台，研究已知单关节硬锁定后的极少样本 Push 恢复；
GenkiArm 与 Panda 仅承担跨机械臂和任务可行性验证。

## 当前入口

- [最新状态与证据边界](LATEST-STATUS.md)
- [当前项目计划（V6）](PROJECT-PLAN-V6.md)
- [严格主结果来源台账](reports/primary-result-provenance-ledger-20260831.md)
- [大幅优势指标机器审计](results/final/large-advantage-metric-audit.json)
- [当前英文主稿](paper/main.pdf)
- [8月30日给学长的完整进度](reports/to-senior-2100-progress-20260830.md)
- [近期顶会项目复现与迁移审计](reports/reproduction-first-audit-20260830.md)
- [SFET 原始臂任务级 No-Go](reports/sfet-task-level-nogo-20260830.md)
- [实际 GenkiArm 三种子阶段结论](reports/genkiarm-three-seed-interim-20260830.md)
- [2025--2026 强相关顶会复现审计](reports/closest-top-conference-methods-2025-2026-20260829.md)
- [最终 G2 证据汇总](reports/g2-final-synthesis-20260821.md)
- [实验进展与结论边界](EXPERIMENT-LOG.md)
- [学长备份审计](reports/senior-backup-audit-20260814.md)
- [G1 Push 正式闸门](reports/g1-push-formal-gate-20260818.md)
- [G1 主动校准诊断](reports/g1-active-calibration-diagnostic-20260818.md)
- [G1 目标导向 Push zero-shot smoke](reports/g1-push-zero-shot-smoke-20260818.md)
- [G1 连续方法审计与最终闸门](reports/g1-overnight-method-audit-20260819.md)
- [G1 Oracle residual 上界](reports/g1-oracle-residual-upper-bound-20260819.md)
- [文档索引](docs/README.md)
- [机械臂当前状态](docs/hardware/robot-status.md)
- [G0 可行性记录](reports/g0-feasibility.md)
- [G1 仿真门禁状态](reports/g1-6dof-provisional-gate.md)

## 目录

```text
config/       实验、模型、环境和不可变数据划分
src/robotarm/ Python 包：环境、模型、训练与数据接口
sim/assets/   MuJoCo 模型、GenkiArm 网格和仿真资产
hardware/     真机关节映射、安全限制和标定记录
scripts/      数据采集、基准实验和烟雾测试入口
tests/        单元测试与接口契约测试
docs/         设计、硬件、计划归档和研究笔记
references/   论文 PDF 与参考文献数据库
papers/       本地文献索引；下载的 PDF/全文不上传公开仓库
reports/      阶段报告和实验门禁结论
results/      最终结果、诊断与复现的小型机器可读输出
runs/         本地实验输出，不纳入 Git
external/     外部完整工程的本地参考归档，不纳入 Git
```

## 快速验证

严格证据环境使用 Python 3.12.10；精确包版本见
`requirements-primary-lock.txt`和
`config/environment/primary-environment-lock.json`。CUDA PyTorch需从官方cu128
索引安装；GPU用于正式评测加速，但CPU/CUDA指标等价性已单独审计，RTX 4060并非
数值正确性的必要条件。

```powershell
python -m pytest
python scripts/run_offline_pipeline.py --help
python scripts/run_g1_benchmark.py --help
python scripts/analyze_seed_significance.py results/final/heldout_5seeds_merged.csv
python scripts/run_push_benchmark.py --seeds 7,17,27,42,51 --epochs 60
```

严格三seed主证据可用一条命令重新汇总并验证。必要的小型原始运行摘要已跟踪；
D3确认候选压缩包缺失时会按冻结seed确定性重建并校验SHA-256：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_primary_evidence.ps1
```

该入口重新生成主结果、决策损失、同容量全局残差、无投影消融、D3确认、环境、
大幅优势指标边界和PDF匿名审计JSON，并运行全部聚焦契约测试。

TD-MPC2 原始臂 adapter 需要可选依赖
`pip install -e ".[tdmpc2]"`，并对单独下载的上游仓库应用
`third_party/patches/tdmpc2-original-arm.patch`。LeWM 复现依赖其官方
`stable-pretraining/stable-worldmodel` 环境与发布 checkpoint；完整版本和
已验证边界见 `reports/reproduction-first-audit-20260830.md`。

## 当前结论（2026-08-31）

历史大幅正向 Push 结果来自简化开发模型，不能作为原始真机、GenkiArm 或闭环
优势。最新原始臂筛选表明，单步 SFET 运输与 hard-mask 在 D2/D3 上没有形成
任务优势；此前称为 `oracle_ik` 的方法实际是不使用隐藏真值的可部署
fault-aware constrained IK 强基线，D2 为 3/3 种子 100%，D3 为
80/100/100%。因此后续方法必须共享其故障可行路径，只比较未知接触效果和
动作排序的极少样本适配。

严格 D2/D4 三 seed、每 seed 400 组、每组 128 候选、50 步评测表明：同容量
全局故障残差相对 nominal WM 的 top-1 regret 平均降低 19.76%（3/3 seed），
终点候选误差降低 4.04%（3/3）；但接触响应 RMSE 恶化 270.04%。移除解析投影
会产生平均 6.63 度锁定关节漂移和 0.539 rad/s 速度违例，启用投影后 3/3 seed
严格为零。

选择性 IPWM 未超过同容量全局残差，full-state 与 selective publication 在当前
协议中完全相同，因此选择性结构归因是 No-Go。当前可支持的是硬约束、动作排序
改善与“预测 RMSE 不等于控制效用”的六阶段诊断；receding-horizon MPC、跨臂
object/contact、方法级 Grasp、真机收益和 4+/5 ICRA 结论仍未成立。

完整过程输出保存在本地 `runs/` 或单独实验备份中；Git 跟踪可复现代码、配置、
审计报告，以及 `results/final/`、`results/analysis/`、
`results/diagnostics/` 和 `results/reproductions/` 的小型聚合结果。
