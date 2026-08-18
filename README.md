# DFWM Robot Arm

面向低成本五自由度机械臂加独立夹爪的关节故障恢复仿真、标定与实验仓库。当前主线以
GenkiArm、MuJoCo 和 Damage-Factorized World Model（DFWM）为核心。

## 当前入口

- [项目计划](PROJECT-PLAN-V5.md)
- [实验进展与结论边界](EXPERIMENT-LOG.md)
- [学长备份审计](reports/senior-backup-audit-20260814.md)
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

Reach 的五随机种子结果已复算，DFWM 相对 topology-only 的差异不显著。Push
任务的多步预测误差改善 15.8% 是当前初步结果；在完成六方法、五随机种子和
bootstrap/置信区间检验前，不作为论文定论。

正式 G1 结果必须在 G0 真机几何、零位、安全范围和 D2/D3 可达性确认后生成。
完整过程输出保存在本地 `runs/` 或单独实验备份中，Git 只跟踪可复现代码、配置和
`results/final/` 聚合结果。
