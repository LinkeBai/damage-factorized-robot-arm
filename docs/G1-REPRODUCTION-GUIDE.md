# G1 复现操作指南

本文用于在 Windows + NVIDIA GPU 环境中复现当前 G1 仿真实验。整个流程默认只运行 MuJoCo，**不连接、不控制实体机械臂**。

## 1. 复现目标

复现以下结果：

- G1 四方法预测实验：topology-only、residual-only、monolithic matched、factorized DFWM；
- D2/D3、3 seeds、K=0/1/2/5；
- IK+PD hybrid baseline；
- Jacobian residual feedback；
- world-model gated hybrid；
- V6 residual-aware option selector；
- G1/V6 汇总表、日志与 manifest。

当前参考提交：`f8ab28e` 或之后的提交。运行前请记录实际 `git rev-parse HEAD`。

## 2. 环境要求

- Windows 10/11；
- Python 3.11 或 3.12；
- NVIDIA GPU，建议 8 GB 以上显存；
- 当前机器 RTX 4060 Laptop 8 GB 已验证可运行；
- 项目目录：`C:\Users\asus\Desktop\robot-arm`。

检查：

```powershell
cd C:\Users\asus\Desktop\robot-arm
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch,mujoco; print(torch.cuda.is_available()); print(mujoco.__version__)"
git rev-parse HEAD
git status --short
```

预期：CUDA 为 `True`。工作区可以有归档文件变化，但不要删除 `sim/`、`src/`、`config/`、`results/final/`。

## 3. 首次安装

若 `.venv` 不存在：

```powershell
cd C:\Users\asus\Desktop\robot-arm
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,plot]"
```

安装后先运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

参考结果为 `110 passed`。

## 4. 一键复现

快速检查，约 1 分钟：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\reproduce_g1.ps1 -Mode quick
```

完整复现：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\reproduce_g1.ps1 -Mode full
```

完整模式包含三 seed 模型训练和 option selector，RTX 4060 Laptop 上可能需要数小时。运行期间不要同时启动其他 G1 Python 进程。

## 5. 分步运行

四方法预测基准：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_g1_benchmark.py --out runs/g1_reproduction --seeds 7,17,27 --epochs 60 --train-trajectories 2 --calibration-trajectories 5 --evaluation-trajectories 3 --trajectory-steps 100 --latent-steps 50 --data-policy controller --device auto
```

Hybrid 基线：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_g1_hybrid_baseline.py
.\.venv\Scripts\python.exe -u scripts\run_g1_residual_feedback.py
```

World-model hybrid：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_g1_worldmodel_hybrid_sequential.ps1
```

V6 option selector：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v6_option_selector_sequential.ps1
```

重新汇总：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_v6_gate.py
.\.venv\Scripts\python.exe scripts\build_g1_manifests.py
```

## 6. 进度查看

World-model hybrid：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\watch_worldmodel_progress.ps1
```

Option selector：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\watch_v6_selector_progress.ps1
```

GPU 状态：

```powershell
nvidia-smi -l 3
```

## 7. 输出位置

- 临时运行目录：`runs/`；
- 最终 CSV/JSON：`results/final/`；
- 报告：`reports/`；
- 日志：`runs/*.log`、`runs/*.err.log`；
- manifest 索引：`results/final/g1-run-manifest-index.json`。

关键结果：

- `results/final/g1-benchmark-20260810/`；
- `results/final/g1-hybrid-baseline.csv`；
- `results/final/g1-worldmodel-hybrid-seed*.csv`；
- `results/final/v6-option-selector-seed*.csv`；
- `reports/g1-final-audit.md`；
- `reports/v6-hybrid-gate.md`。

## 8. 验收与解释

当前可复现结论：

- 原始 learned-MPC 控制门为 No-Go；
- 四方法预测门有窄幅改善；
- IK+PD、world-model hybrid 和 option selector 均可稳定完成 Reach；
- option selector 当前没有超过 IK+PD 的平均步数，因此不能宣称已形成 ICRA 主结果。

不要将 Hybrid 的成功率解释为 world model 的独立增益。论文主张必须以严格消融和统计结果为准。

## 9. 常见问题

`CUDA out of memory`：确认没有重复 Python 实验进程，降低并行任务，不要并行跑多个 seed。

日志长时间不更新：检查 `nvidia-smi`。GPU 有利用率且 Python 进程存在时通常仍在训练；结果多在一个 seed 结束后写入。

PowerShell 中文乱码：不影响 CSV/JSON；运行监控脚本使用 ASCII 输出。

实验中断：保留已有 `runs/`，从未完成 seed 重新运行。不要删除已完成结果；将失败原因写入日志或 exclusion ledger。

## 10. 交接结论

学长只需要先执行快速模式确认环境，再执行完整模式。完成后检查 `results/final/` 和 `reports/`，并将实际 commit、GPU 型号、总耗时和异常情况补入实验记录。
