# 论文清单（28篇，按相关度分四梯队）

标注说明：出处按论文发表venue；CCF等级仅供参考（RSS/CoRL/RA-L不在CCF目录但业内认可度很高）。

---

## 第一梯队：核心必读 —— 低成本硬件 + 模仿学习（你要做的事的直接模板）

| # | 论文 | 出处 | 链接 | 和你的关系 |
|---|------|------|------|-----------|
| 1 | **ACT / ALOHA**: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | RSS 2023 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) · [项目页](https://tonyzhaozh.github.io/aloha/) | **一切的起点**。证明约$20k简易双臂+ACT算法能做开调料杯、插电池等精细任务（成功率80-90%），这类任务传统上需要昂贵力控臂。你的臂上要跑的第一个算法 |
| 2 | **Diffusion Policy**: Visuomotor Policy Learning via Action Diffusion | RSS 2023（期刊版IJRR） | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 和ACT并列的另一大基线，扩散模型生成动作。写论文时必对比的方法 |
| 3 | **Robot Learning: A Tutorial** | LeRobot团队, 2025 | [arXiv:2510.12403](https://arxiv.org/abs/2510.12403) | **为你这种情况写的教程论文**：从零讲到在低成本臂上跑通采数据→训练→部署全流程，当教材精读 |
| 4 | **GELLO**: A General, Low-Cost, and Intuitive Teleoperation Framework | IROS 2024 (CCF-C) | [arXiv:2309.13037](https://arxiv.org/abs/2309.13037) | 3D打印百元级主从遥操作。**给自制臂配leader臂时直接抄它的方案**，也是"低成本硬件发CCF-C"的范例 |
| 5 | **Mobile ALOHA**: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation | CoRL 2024 | [arXiv:2401.02117](https://arxiv.org/abs/2401.02117) | ALOHA+移动底盘做炒菜、擦桌子等长程任务，看"简易硬件能力上限"在哪 |
| 6 | **UMI**: Universal Manipulation Interface | RSS 2024 | [arXiv:2402.10329](https://arxiv.org/abs/2402.10329) | 手持夹爪采演示数据，完全绕开机器人本体。打印件+相机就能复刻的思路 |
| 7 | **Bi-ACT / ALPHA-α**: Position and Force Information for Imitation Learning with Low-Cost System | 2024 | [arXiv:2411.09942](https://arxiv.org/abs/2411.09942) | ACT加入力信息的低成本改进。**示范了"在ACT上做增量创新写论文"的标准姿势** |
| 8 | **SmolVLA**: A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025 | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | 整篇实验就在SO-100（和你同档位的硬件）上做的VLA模型，消费级GPU可跑 |
| 9 | **U-ARM**: Ultra Low-Cost General Teleoperation Interface | ICRA 2025 (CCF-B) | [arXiv:2509.02437](https://arxiv.org/abs/2509.02437) | 超低成本遥操作接口上CCF-B的例子 |
| 10 | **Benchmarking VLA Models on SO-101**: Failure and Recovery Analysis | 2026 | [arXiv:2606.08881](https://arxiv.org/abs/2606.08881) | 在SO-101上系统评测ACT和各VLA模型。**你设计实验和评测协议时的直接参照** |
| 11 | **Data Scaling Laws in Imitation Learning for Robotic Manipulation | 2024 | [arXiv:2410.18647](https://arxiv.org/abs/2410.18647) | 模仿学习要采多少数据、怎么采效率最高的规律，设计数据采集实验前必看 |
| 12 | **Is Your Imitation Learning Policy Better than Mine?** | 2025 | [arXiv:2503.10966](https://arxiv.org/abs/2503.10966) | 评测的统计学规范（试验次数、置信区间）。让你的实验结果经得起审稿人挑战 |
| 13 | **ARMimic**: Learning Manipulation from Passive Human Demonstrations in AR | 2025 | [arXiv:2509.22914](https://arxiv.org/abs/2509.22914) | AR采集人类演示、不需要遥操作硬件——数据采集侧创新的又一条便宜路线 |

## 第二梯队：跨机体迁移 —— "简易臂借用复杂臂的数据和能力"（最贴合你的选题的方向）

| # | 论文 | 出处 | 链接 | 和你的关系 |
|---|------|------|------|-----------|
| 14 | **Open X-Embodiment**: Robotic Learning Datasets and RT-X Models | ICRA 2024 最佳论文 (CCF-B) | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) | 22种机器人的百万条轨迹汇成一个数据集，证明跨机体数据互相有用。**自制臂"白嫖"Franka等昂贵臂数据的理论基础** |
| 15 | **Octo**: An Open-Source Generalist Robot Policy | RSS 2024 | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | 在OXE上预训练的通用策略，几百条演示就能微调到没见过的新机器人——包括你的自制臂 |
| 16 | **OpenVLA**: An Open-Source Vision-Language-Action Model | CoRL 2024 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | 开源7B VLA模型，验证了OXE预训练对全新机体也有提升。微调实验的标准底座 |
| 17 | **π0**: A Vision-Language-Action Flow Model for General Robot Control | Physical Intelligence, 2024 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | VLA路线当前的能力天花板，了解领域走向 |
| 18 | **Latent Action Diffusion for Cross-Embodiment Manipulation** | 2025 | [arXiv:2506.14608](https://arxiv.org/abs/2506.14608) | 潜空间动作表示实现跨机体迁移的具体算法 |
| 19 | **AnyBody**: A Benchmark Suite for Cross-Embodiment Manipulation | 2025 | [arXiv:2505.14986](https://arxiv.org/abs/2505.14986) | 跨机体操作的标准评测集，做这个方向要在它上面报结果 |
| 20 | **Learning Action Priors for Cross-Embodiment Robot Manipulation** | 2026 | [arXiv:2606.26095](https://arxiv.org/abs/2606.26095) | 跨机体动作先验的最新工作 |
| 21 | **KITE**: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation | 2026 | [arXiv:2606.22113](https://arxiv.org/abs/2606.22113) | 运动学与交互解耦→零样本迁移到新机体。对自制臂（运动学和主流臂都不同）尤其相关 |

## 第三梯队：仿真平台与Benchmark —— 零成本做实验

| # | 论文 | 出处 | 链接 | 和你的关系 |
|---|------|------|------|-----------|
| 22 | **LIBERO**: Benchmarking Knowledge Transfer for Lifelong Robot Learning | NeurIPS 2023 (CCF-A) | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) | 130个语言条件操作任务，VLA评测事实标准 |
| 23 | **ManiSkill3**: GPU Parallelized Robotics Simulation and Rendering | 2024（前代NeurIPS） | [arXiv:2410.00425](https://arxiv.org/abs/2410.00425) | GPU并行仿真，训练速度快，支持导入自定义机器人（可以建你自制臂的模型） |
| 24 | **RLBench**: The Robot Learning Benchmark & Learning Environment | RA-L 2020 | [arXiv:1909.12271](https://arxiv.org/abs/1909.12271) | 100个操作任务的经典benchmark |
| 25 | **robosuite**: A Modular Simulation Framework and Benchmark for Robot Learning | 2020 | [arXiv:2009.12293](https://arxiv.org/abs/2009.12293) | MuJoCo基座的模块化仿真框架 |
| 26 | **SAPIEN**: A SimulAted Part-based Interactive ENvironment | CVPR 2020 (CCF-A) | [arXiv:2003.08515](https://arxiv.org/abs/2003.08515) | 关节物体（柜门、抽屉）交互仿真，ManiSkill的底座 |

## 第四梯队：综述（工具书，写Related Work时查）

| # | 论文 | 出处 | 链接 | 用途 |
|---|------|------|------|------|
| 27 | **Imitation Learning in the Deep Learning Era**: A Novel Taxonomy and Recent Advances | 2025 | [arXiv:2511.03565](https://arxiv.org/abs/2511.03565) | 模仿学习全景分类 |
| 28 | **Interactive Imitation Learning for Dexterous Manipulation**: Challenges and Perspectives | 2025 | [arXiv:2506.00098](https://arxiv.org/abs/2506.00098) | 交互式模仿学习综述 |
| 29 | **VLA Models for UAV and Bimanual Manipulation**: A Review | 2026 | [arXiv:2607.06706](https://arxiv.org/abs/2607.06706) | VLA模型综述 |

---

## 和你的选题最直接的组合

"设计算法/跑模型，让简易机械臂完成复杂机械臂的任务"，清单里对应三条已被验证的路：

1. **模仿学习路线**（#1,2,4,7）：ACT/Diffusion Policy + 自采演示数据 → 便宜硬件做精细任务。最成熟，第一篇论文建议走这条。
2. **跨机体迁移路线**（#14-21）：用昂贵臂的公开数据预训练，迁移到你的自制臂。最贴合你的题目表述，创新空间大，难度也更高。
3. **VLA微调路线**（#8,15,16）：拿SmolVLA/Octo/OpenVLA在你的臂上小样本微调。工程量小，适合快速出结果。
