# 项目地图与学习路线

> 写给半年后回来看这个仓库的自己，以及第一次接手的师弟师妹。
> 目标：在两周内从"知道这个项目做过什么"变成"能改动它的任何一部分"。

## 0. 一句话

在一台五自由度机械臂上，**已知某个关节被锁死**的前提下，用一个学到的世界模型预测"推一下方块会发生什么"，据此在一堆候选动作里挑最好的那个，推到目标位置；仿真和真机都跑通了。

三个层次，从下往上：

| 层 | 问题 | 主要代码 |
|---|---|---|
| 环境层 | 锁定的机械臂 + 可推方块，在 MuJoCo 里长什么样 | `sim/assets/arm_push.xml`, `src/robotarm/envs/` |
| 模型层 | 给定当前状态和一个动作，下一步的机械臂和方块在哪 | `src/robotarm/models/` |
| 决策层 | 有了预测，怎么选动作；真机上怎么闭环 | `models/planner.py`, `models/contact_action_ranker.py`, `src/robotarm/deployment/` |

---

## 1. 先解决命名混乱（最重要的一步）

这是全仓库对新人最不友好的地方。同一个东西在四个地方有四个名字：

| 出现位置 | 名字 | 指的是什么 |
|---|---|---|
| Git 仓库名 | `damage-factorized-robot-arm` | 项目最早的名字（损伤因子化世界模型时期） |
| `README.md` 标题 | IPWM Robot Arm | 2026-08-30 改名后的项目名 |
| 论文 (`paper/main.tex`) | **LockPusher** | 对外发表用的方法名 |
| 代码类名 | `BlockTriangularDPWM` (BT-DPWM) | 实际部署的世界模型 |
| 代码类名 | `SelectiveInterventionRollout` (SI) | 论文里的 state isolation / 双分支 |
| 旧文档 | DFWM | 最早的"因子化世界模型"，已被推翻的那一版 |

**对应关系（记住这张表，读代码时会省下大量时间）：**

```
论文 LockPusher  =  TopologySurgery（解析投影）
                 +  BlockTriangularDPWM（世界模型主干）
                 +  SelectiveInterventionRollout（双分支状态隔离）
                 +  contact_geometry（接触几何残差的输入）
```

论文里的 `Carrier` = 代码里 `carrier_model`；论文里的 `Global` = 允许残差改所有坐标的那个变体；论文里的 `Intervention branch` = `intervention_model`。

**建议的整理动作**：在 `README.md` 顶部加一张这个对照表。这是投入产出比最高的一次提交。

---

## 2. 论文 → 代码 对照表

按论文章节顺序读，每读一节去看对应的文件：

| 论文位置 | 内容 | 代码 |
|---|---|---|
| §III | 问题定义、锁定诊断 `d=(m, q̄)` | `envs/damage.py`, `envs/constraint_lock.py` |
| §IV-A 式(4)(5) | 解析锁定投影 Π_d | `models/topology_surgery.py` ← **从这里开始读** |
| §IV-B 式(6)(7) | 为什么必须隔离 rollout 状态 | `models/selective_intervention_rollout.py` 的 docstring |
| §IV-C 式(8)(9)(10) | 双分支、私有 hidden state、选择性读出 | `models/selective_intervention_rollout.py` |
| §IV-D 式(11) | 接触几何残差、stop-gradient | `models/contact_geometry.py`, `models/block_triangular_dpwm.py` |
| §IV-E 式(12) | 归一化状态损失、冻结机器人参数 | `training/g1_mechanism.py` |
| §IV-F Alg.2 | 候选打分与选择 | `models/planner.py`, `models/contact_action_ranker.py` |
| §V-A | MuJoCo 基准、五种锁定 | `sim/assets/arm_push.xml`, `envs/mujoco_env.py` |
| §V-C | 真机视觉闭环 | `deployment/vision_pose.py`, `deployment/real_calibration.py` |

---

## 3. 代码阅读顺序（十个文件，从小到大）

不要按目录顺序读，按下面的顺序。每个文件后面是"读完应该能回答的问题"。

1. **`envs/fk.py`** (131 行) — 正运动学。
   → 关节角怎么变成推杆末端的位置？
2. **`models/topology_surgery.py`** — 解析投影，论文式(4)(5)。
   → 为什么投影两次等于投影一次？状态向量 `[q(5), qvel(5), object(4)]` 的每一维是什么？
3. **`models/contact_geometry.py`** (89 行) — 接触门控。
   → `pusher_box_contact_gate` 返回的是什么？为什么用 sigmoid 而不是硬判断？
4. **`envs/damage.py`** (162 行) — 锁定怎么被施加到环境里。
   → 论文里的 J1–J5 在代码里如何表示？
5. **`models/selective_intervention_rollout.py`** — **全项目最核心的文件**。
   → 两个分支各自维护什么？哪些量被返回给规划器，哪些量喂回递归？
6. **`models/world_model.py`** (221 行) — 世界模型的通用接口。
   → 一次 `step` 的输入输出契约是什么？
7. **`models/block_triangular_dpwm.py`** (713 行) — 部署用的主干模型，最大的一个。
   → 机器人块为什么不消费 object state？stop-gradient 加在哪一条边上？
   → 这个文件不要一次读完，先只读 `forward` 的主路径，把可选参数（那一长串 `reaction_*`）全当 False。
8. **`models/planner.py`** (248 行) — CEM 规划器。
   → 128 个候选是怎么采样的？elites 起什么作用？
9. **`training/g1_mechanism.py`** (972 行) — 训练入口，最长的文件。
   → 六个方法（topology_only / history / param-matched / residual / monolithic / dfwm）分别是什么？
   → 从 `main()` 往下追，不要从头读。
10. **`deployment/vision_pose.py`** — 真机视觉。
    → 图像坐标怎么变成方块的世界坐标？标定参数从哪来？

配套：每读完一个文件，去 `tests/` 找同名测试跑一遍（`pytest tests/test_constraint_lock.py -v`）。**测试是最好的文档**——它写明了每个模块的输入输出契约。

---

## 4. 需要补的背景知识（按用到的地方排，不是按课程排）

只列这个项目真正用到的，每项给一个"够用就行"的判据。

| 主题 | 够用的标准 | 建议材料 |
|---|---|---|
| 刚体变换 / 正运动学 | 能手推五连杆的 DH 或齐次变换链 | 《Modern Robotics》第 3–4 章 |
| MuJoCo 基础 | 能读懂 `arm_push.xml` 每个 tag，能自己加一个物体 | MuJoCo 官方文档 XML Reference |
| 平面推送力学 | 知道为什么推送是欠驱动的、什么是 motion cone | 论文引用 [14] Yu et al. 2016、[15] Zhou et al. 2018 |
| GRU / 序列模型 | 能说清 hidden state 在 rollout 里如何传递 | 任一 RNN 教程 + 亲手实现一次 |
| 图神经网络（消息传递） | 能看懂两轮 message passing 在做什么 | 论文引用 [18] Interaction Networks |
| 世界模型 + 基于模型规划 | 知道 model-based planning 与 model-free RL 的分工 | 论文引用 [1] Finn & Levine, [22] Nagabandi |
| 残差物理 | 知道"解析模型 + 学习修正"为什么比纯学习省数据 | 论文引用 [16][17], [6] ActivePusher |
| 相机标定 / 位姿估计 | 能独立完成一次棋盘格标定 | OpenCV 官方 calibration 教程；`scripts/calibrate_eye_in_hand_camera.py` |

**优先级**：1、2、5 是读代码的门槛，先补；3、7 是理解方法为什么这样设计，第二批；6 是写下一篇论文时的弹药，最后。

---

## 5. 仓库整理建议

现状：`scripts/` 180 个文件，`config/experiment/` 约 200 个 YAML，`src/robotarm/models/` 28 个模型文件，但论文实际只用到其中 4 个。这是一年探索的自然结果，不是错误——但现在该分层了。

建议按这个顺序做，每步一个提交：

1. **标注主线**：在 `README.md` 加上第 1 节的命名对照表 + 第 3 节的十文件阅读顺序。
2. **分离 active / archive**：把被推翻或已停止的方向（DFWM 时期、GenkiArm、Panda、SFET、dual-expert 等）移到 `src/robotarm/models/archive/` 和 `scripts/archive/`，**不要删**——它们是"我们试过什么"的证据，将来写 related work 和答辩都要用。
3. **给主线脚本加前缀**：论文用到的脚本统一改名 `paper_*.py`，一眼可见。
4. **配置瘦身**：`config/experiment/` 里 `z0`–`z81` 系列是探索留下的，归档到 `config/experiment/archive/`，保留论文用到的那几个。
5. **写一个 `docs/RESULTS-INDEX.md`**：论文里每个数字 → 产生它的脚本 + 结果 JSON 路径。将来任何人问"这个 76.53% 哪来的"，一行就能答。

第 5 条是所有条里最重要的。

---

## 6. 这个项目已经教给你的东西（别丢掉）

翻 `EXPERIMENT-LOG.md` 和 `LATEST-STATUS.md`，里面有几条方法论，比任何一个模型都值钱：

1. **三个种子的结论是不可信的。** 最初 3 seeds 显示 DFWM 显著优于基线，5 seeds 之后优势完全消失——记录里写的是"假象，被推翻"。以后任何结论先跑到 5 seeds。
2. **先定闸门，再看结果。** 项目里有大量"preregistered gate"（如"至少 10% regret reduction 且 3/3 方向一致"），并且在没过闸门时如实记为 No-Go。这是这个项目最专业的部分。
3. **区分"预测变好"和"决策变好"。** 六阶段诊断（约束 → 可达 → 接触 → 响应预测 → 动作排序 → 实际结果）发现过"响应 RMSE 变差但动作选得更好"的情况。这两件事不是一回事，是这个项目最有价值的发现之一。
4. **负面结果要留在记录里。** 所有 No-Go 都还在仓库里，没有被悄悄删掉。

这四条你已经在实践中学会了。很多做完博士的人也未必。

---

## 7. 两周计划（建议）

| 时间 | 做什么 | 产出 |
|---|---|---|
| 第 1–2 天 | 第 1 节命名表 + 第 2 节对照表写进 README | 一个提交 |
| 第 3–6 天 | 按第 3 节读十个文件，每个跑对应测试 | 每个文件写 5 行笔记 |
| 第 7–8 天 | 亲手改一个东西：换个物体尺寸重跑一次仿真评估 | 确认自己能动它 |
| 第 9–10 天 | 补第 4 节优先级 1、2、5 的背景 | — |
| 第 11–14 天 | 做第 5 节的归档整理 + `RESULTS-INDEX.md` | 三到四个提交 |

两周之后，这个仓库会从"一年堆出来的东西"变成"我能讲清楚并且能改的系统"。这个状态本身，就是找导师、申请、面试时能直接拿出来的东西。
