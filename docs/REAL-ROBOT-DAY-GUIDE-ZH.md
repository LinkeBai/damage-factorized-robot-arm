# 原始 5-DoF 机械臂真机实验：零基础现场手册

日期：2026-09-01  
适用范围：原始 5-DoF 夹爪机械臂、一个方块、两个固定眼在手外摄像头。  
今天的正式目标：收集 Level-A 固定低速轨迹的物理可行性证据，不比较学习模型。

现场实际设备已核验为：大恒 `MER2-230-168U3C`（序列号
`FDE23080341`）作为俯视工业相机，`icspring camera` 作为水平相机；普通
`USB2.0 HD UVC WebCam` 不是实验相机。大恒相机必须通过 Galaxy SDK 访问，
不能把它当成普通 OpenCV/DirectShow 摄像头编号。

## 1. 用一句话理解项目

机械臂某个关节被锁住后，普通世界模型可能继续预测这个关节会运动，也可能预测
误差变小却选不出真正有效的推块动作。项目研究如何加入故障硬约束、用极少故障
数据适配世界模型，并逐层检查“预测改善为什么能或不能变成控制收益”。

目前仿真已经支持两条可靠结论：解析投影可把锁定关节的测量违例降到零；同容量
全局故障适配在 D2/D4 开发集上将 response RMSE 平均降低 26.45%（3/3 seeds）。
后者不是选择性 IPWM 的独占优势，控制结果也只有 2/3 seeds 同向。今天的真机
不能宣称验证了学习方法优越性，因为仿真动作是广义力，真机接口是舵机目标位置，
二者尚无验证过的动作映射。

## 2. 三种实验条件

- intact：机械臂正常，没有锁定关节。
- D2：第二关节锁定。整个动作中 J2 指令必须保持同一个角度。
- D3：第三关节锁定。整个动作中 J3 指令必须保持同一个角度。

“锁定”不是把该关节命令设为零，而是让它保持在预先记录的固定角度。实验中还要
测量实际关节与固定角度的偏差。安全审计的最大允许锁定漂移为 3.5 度。

## 3. 今天每个人的职责

负责人：连接和控制机械臂，验证关节方向、限位、急停和三条低速轨迹；决定是否
立即中止。零基础同学不得独自修改控制器、关节限位或故障实现。

协助同学：按标记摆放方块；检查两台相机是否录制；报读试验编号和条件；填写每行
结果；记录失败原因；每十次检查文件；结束时完成两份备份。

任何人发现电缆拉扯、机械臂越界、异常抖动、意外碰撞、相机掉线或无法随时停止，
立即说“停”，负责人停止动作。不要为了补一次成功继续冒险。

## 4. 今天真正需要得到什么

正式 Level-A 包含 30 次 Push：intact、D2、D3 各 10 次。每次都执行对应条件下
同一条已经验证的固定低速位置轨迹。方块每次回到同一物理标记，目标点固定。

每次必须保存：

1. 头顶/斜上方固定相机视频；
2. 水平方向固定相机视频；
3. 带时间戳的五关节位置、速度和控制命令日志；
4. 是否到达、是否接触、方块终点误差、是否成功；
5. 最大锁定误差；
6. 若中止，保留该行并填写 failure_code。

成功阈值冻结为方块终点距离目标不超过 0.03 m。接近接触阈值为 0.01 m。不要看
完结果后修改阈值。两个摄像头都在机械臂外，必须称为 eye-to-hand，不能写成
eye-in-hand。

## 5. 开机前布置

1. 清空机械臂工作区，固定底座和桌面；急停必须伸手可及。
2. 用胶带标记机械臂底座、方块初始位置、目标位置和相机脚架位置。
3. 固定头顶/斜上方相机和水平相机；整个正式试验中不得移动。
4. 固定线缆，保证五个关节走完整条轨迹时不拉紧、不卷入夹爪。
5. 拍摄全景照片：机械臂、夹爪、方块、桌面、两台相机、标定板、急停和线缆。
6. 记录机械臂、夹爪、方块资产编号以及两台相机序列号。
7. 建立 left_video、horizontal_video、control_log、backup_1、backup_2 文件夹。

建议文件名包含试验顺序和条件，例如：
`trial_001_intact_left.mp4`、`trial_001_intact_horizontal.mp4`、
`trial_001_intact_control.csv`。文件名必须与完成日志中的路径完全一致。

## 6. 相机与时钟同步

1. 两台相机完成各自的标定，保存标定文件，不能只口头说“标过”。
2. 两台相机和控制日志同时开始记录。
3. 在两台画面都能看见的位置做一次清晰同步事件，例如快速开关标定板或闪光；
   同时在控制日志记录事件时间。
4. 会话结束时再做一次同步事件，检查整场有没有时间漂移。
5. 允许的最大同步误差为 50 ms。相机中途移动或掉线后，停止正式试验并重新标定/
   同步；不能把前后两段默认当成同一坐标系。

## 7. 先验证机械臂，不要直接跑正式试验

由负责人完成：

1. 单关节、小幅度、低速点动 J1--J5，核对实际运动方向和软件编号一致。
2. 验证软件停止和实体急停；任何一个不能可靠停止都不得继续。
3. 核对实测关节限位。
4. 所有固定轨迹的命令速度不得超过 5 度/秒。
5. 分别在 intact、D2、D3 下运行不接触方块的低幅度探针。
6. D2 中确认 J2 命令全程恒定；D3 中确认 J3 命令全程恒定。
7. 在三种条件下各手工验证一条可到达并能安全推块的固定轨迹。先单步/短段运行，
   再完整运行。轨迹未经验证，不得填写一个看似正式的 trajectory_id。

仓库的舵机协议默认使用 `COM3`、1,000,000 baud；Windows `mode COM3` 显示的
19200不是项目程序实际打开串口时的速率。开始运动前先运行只读探针并确认1--5号
舵机全部返回位置和温度：

```powershell
.\.venv-cuda\Scripts\python.exe scripts\probe_servo_bus_readonly.py `
  --port COM3 --baudrate 1000000 `
  --output results\real_robot\servo-bus-readiness.json
```

该工具不写寄存器、不使能扭矩。只读探针未PASS时，禁止运行任何运动脚本。

每条轨迹至少包含两个时间点，时间从0开始严格增加，waypoint编号从0连续编号，
并以弧度保存五关节位置。CSV列为：

`trajectory_id,condition,waypoint_index,time_s,j1,j2,j3,j4,j5`

## 8. 生成并冻结正式顺序

三条轨迹验证后，为它们取真实ID，例如日期加条件，不要使用含“最佳”“成功”等
结果暗示的名称。然后在仓库根目录运行：

```powershell
.\.venv-cuda\Scripts\python.exe scripts\generate_real_robot_level_a_schedule.py `
  --intact-trajectory-id <真实intact轨迹ID> `
  --d2-trajectory-id <真实D2轨迹ID> `
  --d3-trajectory-id <真实D3轨迹ID> `
  --output data\real_robot\level_a_schedule_frozen.csv
```

接着审计轨迹库：

```powershell
.\.venv-cuda\Scripts\python.exe scripts\audit_level_a_trajectory_library.py `
  data\real_robot\level_a_trajectory_library.csv `
  data\real_robot\level_a_schedule_frozen.csv `
  --output results\real_robot\trajectory-library-audit.json
```

只有输出 `TRAJECTORY_LIBRARY_SAFE_TO_FREEZE` 且退出码为0才可继续。复制冻结表为
`level_a_trials_completed.csv`，之后只填写测量/文件/失败字段；冻结表本身永远不改。

使用真实资产、相机、标定和目录生成会话清单。先运行
`scripts/prepare_real_robot_level_a_session.py --help`，按现场真实值填写所有必填项。
生成后不要手工修改哈希。随后运行 Level-A preflight；只有输出
`LEVEL_A_TRIALS_MAY_START` 才进入正式试验。

## 9. 每一次Push的标准动作

协助同学按以下顺序逐字核对：

1. 报出冻结表中的 `trial_order`、condition、trajectory_id。
2. 确认机械臂处于安全初始位，锁定条件正确。
3. 用复位夹具把方块放回同一胶带标记；不要凭感觉放。
4. 确认目标标记未移动，两台相机正在录制，控制日志已创建。
5. 做一次画面可见的trial开始标记。
6. 负责人执行冻结的固定低速轨迹；协助同学手靠近急停但不进入工作区。
7. 动作停止后不要立即移动方块；先让两台相机多录2--3秒。
8. 记录是否到达推块区域、是否发生接触、最大锁定误差和方块终点位置。
9. 根据冻结的3 cm阈值计算/记录 endpoint_error_m 与 success。
10. 停止并保存两个视频和控制日志；立即确认三个文件能打开且非空。
11. 在 completed CSV 的对应行填写结果和三个相对/绝对文件路径。
12. 方块复位，进入下一行。不得跳到自己喜欢的条件，也不得重排。

若试验异常：立刻停止，`aborted=1`，保留视频和日志并填写 failure_code，例如
`camera_loss`、`workspace_exit`、`cable_risk`、`unexpected_contact`、
`joint_limit`、`emergency_stop`、`controller_error`。不要删除这一行，也不要用补跑
覆盖；若确需额外重跑，作为带说明的额外行保存，正式冻结30行仍原样保留。

## 10. 每5次和每10次检查

每5次：确认两相机视角未动、文件数正确、控制日志可打开、方块复位标记未动。

每10次：把目前所有原始文件复制到 backup_1；随机打开一个双视频和一个控制日志；
核对 completed CSV 没有缺行、重复编号或改动 condition/trajectory_id。

不要在现场删除失败视频来节省空间。磁盘不足就停止并转存，不要继续产生无证据试验。

## 11. 收集完成后的唯一正式流水线

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run_real_robot_level_a_pipeline.ps1 `
  -Manifest data\real_robot\session_20260901.yaml `
  -FrozenSchedule data\real_robot\level_a_schedule_frozen.csv `
  -TrajectoryLibrary data\real_robot\level_a_trajectory_library.csv `
  -CompletedLog data\real_robot\level_a_trials_completed.csv
```

它依次检查轨迹安全、现场清单、冻结表未被篡改、原始文件存在、每条件至少10条有效
数据，并自动生成论文图表。任何一步失败都先修复真实数据/路径问题；禁止单独运行
图表脚本绕过失败门禁。最后把整个会话包完整复制到 backup_1 和 backup_2。

## 12. 可选夹取实验

只有30次Push和两份备份完成后才做。使用同一个方块、固定预抓取位姿、简单闭合
夹爪、最短安全垂直抬升，并保持3秒。intact/D2/D3每种最多5次，所有失败保留。
它只证明固定抓取的物理可行性，不能声称是学习抓取、故障重规划或模型优势。

## 13. 今天绝对不能做的事

- 不把固定轨迹命名成 nominal、global_matched 或 IPWM 方法结果。
- 不在没有动作接口桥的情况下做 Level-B 学习方法优越性声明。
- 不移动相机后继续沿用旧标定。
- 不删除失败、中止或看起来“不好看”的试验。
- 不为了提高成功率改变方块位置、目标、阈值、轨迹或条件顺序。
- 不让零基础同学独自操作关节故障、限位和急停。
- 不把双眼在手外相机写成手眼相机。

## 14. 离场前核对清单

- [ ] intact/D2/D3各10条有效Push，或明确记录不足原因。
- [ ] 所有abort和failure保留。
- [ ] 每条都有双视频、控制日志和填写完整的CSV行。
- [ ] 冻结schedule与completed log分别保存。
- [ ] 标定文件、同步视频、轨迹库、审计JSON和manifest齐全。
- [ ] 正式流水线运行；PASS或失败原因已保存。
- [ ] backup_1与backup_2均能打开，不只是空文件夹。
- [ ] 夹取结果与Push结果分开，未越界声称学习方法优势。
