# Robot-Arm 项目进度备忘（供会话重建时续接）

> 依据学长 **PROJECT-PLAN-V4.md**（DFWM：Damage-Factorized World Model，低成本机械臂关节锁定后的少样本安全恢复）。
> 本文件记录当前进度、环境与下一步，供你重启会话后快速续接。

## 最新进度（2026-08-07）

- 新计划 `PROJECT-PLAN-V4.md` 已冻结六自由度主线。
- 完整 GenkiArm STL 模型包含 7 个 mesh，并恢复 J1-J6 六个执行器。
- `MujocoArmEnv(model_variant="simple"|"mesh")` 均提供统一 6-DoF API。
- J6 定义为夹爪整体姿态轴；夹爪开合若存在则为独立执行器。
- D2/D3/D4 已纠正为锁定 J2/J3/J4，并接入共同可达域分析。
- 新增数值 IK；仿真采集现在使用正确的 `(state, action, next_state)` 时序。
- DFWM 上下文已修复为 `[e_topology(64), z_residual(8)]`，部署适应只更新 `z`。
- `scripts/run_offline_pipeline.py` 已打通模型加载、可达域、轨迹存储、世界模型训练和残差适应。
- 默认离线 smoke：共同可达体素 427；残差 NLL `-7.61 -> -12.10`。
- 旧 5-DoF G1 3-seed 结果已标记 superseded，仅保留为历史预实验。
- 六维 mask/action、十二维 state、J1-J6 mapping 和 provisional target split 已落地。
- conditional RSSM、四方法预测 smoke 与冻结 CEM-MPC 已接通。
- 当前六自由度 G1 gate 为 **NO GO**：D3 frozen-MPC smoke 仍为 0/2 success。
- G0 仍未通过：真机零位、方向、限位、FK 误差、急停、电流和温度均待测。

运行完整离线流程：

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_pipeline.py
```

默认结果目录：`runs/offline_pipeline/<timestamp>/summary.json`。

## 历史进度（截至 2026-08-04，已被六自由度计划覆盖）

- ✅ Git 仓库已在 `robot-arm/` 建立，首次提交 `c1e9d6a`（V4 骨架 + protocol）
- ✅ 环境：`.venv`（Python 3.12）+ GPU torch `2.11.0+cu128`（cuda=True，4060 实测通过）+ mujoco 3.8.1 + lerobot 0.6.1 + gym-aloha/gym-pusht（备用）
- ✅ 目录骨架（§9）：`src/robotarm/{envs,models,training,baselines,data,analysis}`、`config/`、`sim/assets/`、`tests/`、`scripts/`
- ✅ `src/robotarm/envs/protocol.py`：RobotEnv 统一接口（sim/real 共用）
- ✅ `pyproject.toml`：editable 安装（`import robotarm` 可用）+ pytest dev 依赖
- ✅ `src/robotarm/data/schema.py`：§10.1 轨迹 schema（Episode/StepRecord + 校验）
- ✅ `src/robotarm/envs/damage.py`：DamageConfig（joint_mask + lock_angle）、D0..D4
- ✅ `sim/assets/arm.xml`：5-DoF 臂（j1 yaw + j2/j3/j4 pitch + j5 roll，ee site），nq=5，可加载/step
- ✅ `src/robotarm/envs/mujoco_env.py`：MujocoArmEnv（实现 RobotEnv 协议 + damage 锁关节注入）
- ✅ `src/robotarm/envs/fk.py`：解析 FK，与 MuJoCo ee site 误差 ~1e-16（机器精度）
- ✅ `src/robotarm/envs/safety.py`：SafetyMonitor（软限位/速度/ctrl 上限/锁关节禁动/急停决策，阈值来自 G0 而非硬编码）
- ✅ `src/robotarm/envs/tasks.py`：ReachTask / PushTask / PickTask + reward/success
- ✅ `src/robotarm/data/storage.py`：§10.2 追加式存储（episode→不可变 npz、manifest、sha256、exclusion ledger、clean_version 生成新 dataset）
- ✅ 测试 **62 项全过**（schema / damage / fk / env / safety / storage / tasks）

## 下一步（G0 的前置 + 可离线推进）

**必须先做（阻塞任务冻结）：**
1. 向学长确认能否拿到真机数据/委托测量 —— **G0 是 08-09 截止 gate**
2. `hardware/arm_spec.yaml`（G0 交付物）：连杆尺寸、零位、限位 —— 需真机
3. 依据实测替换 `arm.xml`/`fk.py` 的**占位**连杆长度（当前：base 0.06 / 上臂 0.20 / 前臂 0.17 / 工具 0.10）
4. `hardware/safety_limits.yaml`：供 safety.py 读取真实阈值（G0 测量值）

**可离线推进：**
5. `reachability.py`（G0 §4：intact/damaged 共同可达域 + target 采样）
6. `config/` 各实验配置 + `storage` 示例脚本
7. 数据采集闭包（`MujocoArmEnv` + `tasks.py` + `storage.py` 串起来的 collect 脚本）

## 环境使用

```bash
cd C:/Users/asus/Desktop/robot-arm
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 重要：旧中文空壳文件夹

`C:\Users\asus\Desktop\具身智能-简易机械臂-论文清单` 内容已迁到 robot-arm 且已入库。
该空壳文件夹被 CWD 占用，需**关闭本会话后手动删除**。

## 待确认（G0 需要，决定 arm.xml/FK 建模）

- 实体臂是否已有 / 可否真机测量（计划书 G0 依赖）
- 6-DoF 臂构型、J6/TCP 与连杆尺寸（CAD/3D 模型）
- 舵机型号是否确为 Feetech STS3215（影响 MJCF 限位）
