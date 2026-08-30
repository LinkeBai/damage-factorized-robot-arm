# DFWM Robot Arm

面向低成本五自由度机械臂加独立夹爪的关节故障恢复仿真、标定与实验仓库。当前主线以
GenkiArm、MuJoCo 和 Damage-Factorized World Model（DFWM）为核心。

## 当前入口

- [最新状态与证据边界（2026-08-30）](LATEST-STATUS.md)
- [当前项目计划（V6）](PROJECT-PLAN-V6.md)
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
reports/      阶段报告和实验门禁结论
results/      可提交的最终聚合结果
runs/         本地实验输出，不纳入 Git
external/     外部完整工程的本地参考归档，不纳入 Git
```

## 快速验证

```powershell
python -m pytest
python scripts/run_offline_pipeline.py --help
python scripts/run_g1_benchmark.py --help
python scripts/analyze_seed_significance.py results/final/heldout_5seeds_merged.csv
python scripts/run_push_benchmark.py --seeds 7,17,27,42,51 --epochs 60
```

## 当前结论（2026-08-30）

历史大幅正向 Push 结果来自简化 `arm_push.xml`，不能作为实际 GenkiArm 证据。
在校准运动学 GenkiArm 上，预注册种子 107/117/127 的 routed selective SI-IPWM
目标物 RMSE 改善为 +0.6037%、-0.7836%、+0.7182%，平均仅 +0.1794%，置信区间
跨零，当前性能门为 **No-Go**。三个种子均保持自由状态回归为零和锁定坐标违例为零，
因此只支持解析约束与选择性状态隔离的窄机制主张。闭环控制优势、Panda 对象/接触传播、
真机和 4+/5 ICRA 结论均未成立。完整边界见 `LATEST-STATUS.md`。

完整过程输出保存在本地 `runs/` 或单独实验备份中，Git 只跟踪可复现代码、配置、
审计报告和 `results/final/` 聚合结果。
