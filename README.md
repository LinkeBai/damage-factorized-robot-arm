# DFWM Robot Arm

面向低成本五自由度机械臂加独立夹爪的关节故障恢复仿真、标定与实验仓库。当前主线以
GenkiArm、MuJoCo 和 Damage-Factorized World Model（DFWM）为核心。

## 当前入口

- [当前项目计划（V6）](PROJECT-PLAN-V6.md)
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

## 当前结论（2026-08-21）

G0、G1 和 G2 已完成。原始 DFWM residual-identification 分支、CR-GWM/RC-GWM
稳定预测主张以及完整 FT-GWM Push world model 均未通过冻结门禁。当前受到五
随机种子证据支持的主线是 ordinary ensemble averaging 与 selective prediction：
三成员 ensemble 相对参数匹配单模型的多步 RMSE 平均改善 30.74%，50% coverage
下选择性拒绝使 RMSE 降低 50.50%。Topology conditioning 相对 ordinary ensemble
的独立优势不显著；FT-GWM K1 仅保留为满足已知锁定约束的 provisional joint-dynamics
结果。正式真机 G3 尚未启动，下一决策是收缩后的论文主张是否足以支持真机重复验证。

完整过程输出保存在本地 `runs/` 或单独实验备份中，Git 只跟踪可复现代码、配置、
审计报告和 `results/final/` 聚合结果。
