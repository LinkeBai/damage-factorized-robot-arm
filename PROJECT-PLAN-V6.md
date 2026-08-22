# Project Plan V6 — Robust Zero-Shot Structured Dynamics

**项目**：低成本机械臂关节锁定后的稳健零样本结构化动力学
**版本日期**：2026-08-22
**规划模式**：standard  
**规划基线**：本文件是后续执行的最新基线；旧版计划和失败实验保留为审计记录  
**当前状态**：主线已冻结为 **Stable Uncertainty-Calibrated BT-DPWM（Z69+Z70）**。已知锁定拓扑的解析投影、自由关节 residual innovation、robot→independent-object 单向因果链和 K=0 严格零旁路均保留。经过 Warp 数据语义修复、已知拓扑/held-out residual 协议校正、amortized physical-context posterior 和未训练 topology 输入列修复，seed 7/17/27 的 0/5/10/25/50-transition 公平门控全部满足 BT 自身无负迁移、总体随预算单调改善、constraint violation 为零；K10/K25 样本效率超过相同 adapter/数据/预算的 shared h136/240，K50 总体近乎持平。下一阶段为真机 D2/D3 与 object expert 风险边界。
**证据约束**：实测结果均指向可追溯 artifact；未来时间、GPU-h、工时和阈值仍属于项目管理估计

> ## 2026-08-22 Z65 三种子冻结门控（当前最新状态）
>
> - 核心方法为 **Uncertainty-Calibrated Block-Triangular Damage-Projected World
>   Model**：解析投影固定已知锁定关节；residual innovation 只修正自由关节；校准后的
>   robot transition 单向驱动 independent object expert；K=0 时 context 严格为零，
>   不存在静态 topology/residual 旁路。
> - Z63 oracle 在四个 held-out residual 域相对 BT 自身 K0 分别证明约
>   `+9.78%/+19.15%/+1.55%/+11.27%` 的表达上限；Z64/Z65 将仿真训练期物理描述符
>   蒸馏为仅使用 state/action/known-lock-mask 的可真机 amortized posterior，并加入
>   nested-budget consistency、逐轴不确定度、support-validation rollback、后验精度融合、
>   topology observability wait 和 replacement hysteresis。
> - seed 7 四域平均 BT自身增益随预算为
>   `0/3.45/4.78/5.77/5.77%`；seed 17 为
>   `0/0/0/7.79/10.98%`；seed 27 为
>   `0/0/0/2.03/7.70%`。全部 seed×domain×budget 无负迁移，每个 seed 最终至少
>   3/4 域为正，constraint violation 最大值为 `0`。object RMSE 的最大绝对变化仅
>   `0.173%`，且只能来自校准 robot rollout 链路。
> - 权威机器汇总：
>   `runs/g2_bt_dpwm_context_encoder_z65/three_seed_gate_v1/summary.json`；生成器：
>   `scripts/summarize_bt_dpwm_z65_gate.py`。核心测试当前 `16 passed`。
> - **必须诚实保留的风险**：三 seed 的 Z65 适配机制稳定，但 K0 BT 基座相对 shared
>   h136/240 方差仍大。seed 17 overall 明显落后 shared；seed 27 free/overall 更好但
>   object 更差。因此当前结论是“部署适配机制三 seed PASS”，不是“完整模型已全面超过
>   shared”。下一步只在同一 BT-DPWM 主线内稳定基座 robot/object 训练与 checkpoint
>   selection，再做同 adapter/数据/预算的公平 shared 比较。
> - 基座稳定化诊断：Z66 直接从同 seed shared checkpoint 初始化 robot 并对 object
>   validation-best selection，使最差 seed17 overall 劣势从 `-27.96%` 缩小到
>   `-9.18%`，证明 scaffold 初始化是主要方差源之一；Z67 进一步移除内部
>   topology/contact 条件后 overall 灾难性退化 `-281.36%`，明确 **NO-GO**。因此
>   recurrent robot hidden 必须保留已知锁定拓扑，后续不得再以 damage-agnostic 为由
>   删除 topology；只优化同一结构的初始化与验证选择。
> - 真机接口冻结为：已知锁定关节编号/锁定角、ST3215 电流与位置/速度、眼在手上视觉
>   object pose、0/5/10/25/50 条安全 excitation transitions；不要求力传感器。优先 D2、
>   D3 两种锁定条件，每种至少三次安全校准与重复 rollout。

> ## 2026-08-22 Z69+Z70 公平基座与 shared 门控（取代上节“基座待稳定”）
>
> - 根因修复：shared backbone 训练时 topology mask/angle 恒为零，旧 BT 却向对应未训练
>   随机输入列写入真实拓扑，造成 seed-dependent robot 漂移。Z69 从同 seed shared
>   checkpoint 完整复制 robot block，清零这两列，只由解析 state/action projection 使用
>   已知锁定拓扑。三 seed 相对 shared 的 K0 free-arm 回退压到最多 `1.23%`。
> - 在相同 adapter 参数、67 个训练域、context encoder、calibration transitions 和安全
>   acceptance 预算下，Z70 三 seed×四域平均 BT自身增益随 K=0/5/10/25/50 为
>   `0/0.21/1.38/3.53/7.24%`；shared 为 `0/0.20/0.63/2.78/6.45%`。
>   BT 相对 shared 为 `-0.73/-0.72/+0.04/+0.03/-0.19%`：K10/K25 样本效率略优，
>   K50 近乎持平，而 BT 锁定坐标 violation 始终严格为 `0`。
> - 所有 BT seed×domain×budget 相对自身 K0 均无负迁移；aggregate 曲线单调；object
>   RMSE 最大绝对变化仅 `0.113%`，只能经 calibration robot transition 单向影响。
>   机器门控 `passed=true`：
>   `runs/g2_bt_dpwm_z69_adapter_z70/three_seed_fair_gate_v1/summary.json`；报告：
>   `reports/g2-bt-dpwm-z70-fair-three-seed-gate-20260822.md`。
> - 当前可以主张：BT-DPWM 在相同 few-shot 预算下达到 shared 的预测水平、具有更好的
>   中低预算平均适配效率，并提供 shared 不具备的解析零违例与可审计因果结构。不得主张
>   每个单域都胜 shared，也不得掩盖 independent object expert 在 K0 的约 8%--21%
>   相对回退；后者作为真机视觉噪声下的风险边界继续验证。
> - 真机软件闭环已就绪：`scripts/collect_bt_dpwm_real_calibration.py` 默认 dry-run，只有
>   显式 `--execute`、急停确认字符串和新鲜视觉 pose 才允许串口运动；实时监控 current
>   raw、温度、锁定漂移和视觉时间戳，异常顺序 torque-off。输出 14维 state/5维 action
>   后由 `scripts/infer_bt_dpwm_real_context.py` 使用冻结 Z69/Z70/Z65 做相同预算的安全
>   posterior inference，不访问仿真 privileged residual label。当前相关测试 `20 passed`。
>   真机数值结果尚未采集，必须在实体臂连接后完成 D2/D3×3 repetitions，严禁以 dry-run
>   或 synthetic interface test 代替。

> ## 2026-08-21 Q0-B 最终修订（当前最新状态）
>
> - Q0-B 在 held-out D3 主域按 rollout depth 分层，比较 object ensemble
>   disagreement 与其加上 cross-expert joint discrepancy 的等权秩风险分数；不学习
>   权重，目标误差为融合模型 overall RMSE。
> - seed 7/17/27/37/47 的 fixed-depth AURC 改善分别为
>   `18.93%/11.23%/6.46%/15.01%/0.57%`；partial Spearman 分别为
>   `0.583/0.573/0.522/0.551/0.104`。平均改善 `10.44%`，且五个条件相关均为正，
>   但只有 **3/5** seeds 达到单种子 10% 门槛。
> - 按预注册要求至少 4/5 seeds 通过，Q0-B 最终为 **NO-GO**。不得用均值超过
>   10% 覆盖种子稳定性失败，不进行事后调权、改 coverage 或追加 MPC。
> - Q0-A 仍保留为工程正结果；它证明异构专家可以无损组合并满足精确约束，但不足以
>   支撑 DE-DWM 作为论文核心风险机制。权威报告：
>   `reports/g2-dual-expert-gate-q0b-20260821.md`。
> - Q0-B 后只读机制诊断进一步发现：在相同 fused state 上，FT structural expert
>   的局部 joint correction gain 五种子均值为负，正修正样本仅约 `11%--34%`；
>   `u_cross` 也不能稳定预测 correction gain 或 object residual。因此 Q0-A 的多步
>   free-arm 收益更可能来自 exact constraint/geometry 对 recurrent rollout 的流形稳定，
>   而非逐步专家更准。该解释目前仅为探索性诊断，下一候选若继续，必须以 direct
>   projection、matched unconstrained joint expert 和 teacher-forced rollout 做新的归因门。
>   详见 `reports/g2-dual-expert-mechanism-diagnostic-20260821.md`。
> - 新机制随后执行 S0/S1 归因门。FT 相对 ordinary direct projection 在 seed
>   7/17/27 的 depth-10 free-arm 改善为 `55.60%/43.74%/54.22%`，且相对 ordinary
>   的收益从 depth 1 到 depth 10 扩大 `23.40/29.50/23.33` 个百分点，支持广义的
>   recurrent manifold stabilization。但 projected matched graph joint expert 在
>   seed 17/27 又比 FT 好 `42.85%/30.98%`；因此 FT-specific attribution 仅 1/3
>   通过，已不可能达到冻结 4/5，seed 37/47 按停止规则不再运行。结论为：单纯 direct
>   projection 不够，独立 joint expert + projection 有效，但固定变换几何的稳定独立
>   贡献仍不成立。详见 `reports/g2-manifold-stabilization-gate-s0-s1-20260821.md`。
> - 随后的 T0 实现显式 tangent-normal transition：保留完整链空间消息，但清零锁定
>   节点 recurrent hidden 和 joint delta。seed 7 depth-10 free-arm RMSE 为
>   projected matched `0.44`、topology-projected `0.49`、tangent `0.51`；tangent
>   相对两对照回退 `17.14%/4.69%`，判 **NO-GO**，不扩种子。这说明锁定节点的
>   temporal feature 仍携带有用历史，而 exact output projection 已足以消除法向坐标
>   误差；简单 hidden masking 不能构成新核心创新。详见
>   `reports/g2-tangent-manifold-gate-t0-20260821.md`。
> - U0 最终拆解 DPP-WM 主体：seed 7 depth-10 free-arm RMSE 为 monolithic
>   internal projection `0.93`、product no projection `0.44`、product output-only
>   projection `0.44`、DPP internal projection `0.44`。DPP 相对 monolithic 改善
>   `52.86%`，但相对两种 product 对照均仅改善 `0.14%`，核心“internal projection
>   对预测稳定性不可替代”判 **NO-GO**。当前确认的预测收益来自 joint/object 独立
>   transition；projection 的确认价值是把 violation 从 `0.30` 降到 `0`，而不是提高
>   free/object RMSE。详见 `reports/g2-dppwm-core-ablation-u0-20260821.md`。
> - V0 最终消除架构/预算混杂：shared parameter-matched graph（338,102 参数，
>   60轮）、shared compute-matched graph（169,542 参数，120轮）与 independent
>   joint/object graphs（合计339,084参数，各60轮）使用相同数据、优化器和 rollout
>   loss。seed 7 depth-10 free/object 为 shared-param `0.34/0.05`、shared-compute
>   `0.32/0.05`、independent `0.36/0.04`；independent free 分别回退
>   `3.26%/9.82%`，虽 object 改善 `3.94%`，仍判 **NO-GO**。因此早期约53%的
>   product收益不能归因于 factorization，主要混入了 graph joint specialist 与旧 generic
>   ensemble 的架构/训练差异。按冻结顺序不进入 BT-DPWM；混合 backbone 双专家仅保留
>   为工程 incumbent，不作为因果创新证据。详见
>   `reports/g2-dual-expert-fair-gate-v0-20260821.md`。
> - W0 使用 V0 冻结权重做零训练 asymmetric stitch：joint 取 shared-compute
>   graph，object 取 independent object specialist。seed 7 depth-10 free/object/
>   overall 为 shared `0.32/0.05/0.25`、independent `0.36/0.04/0.27`、asymmetric
>   `0.32/0.04/0.24`。asymmetric 相对 shared 仅改善 free `1.32%`、object
>   `3.43%`、overall `1.34%`，未达冻结 `10%/5%` 门槛，判 **NO-GO**，不扩
>   种子或调路由权重。它相对 fully independent overall 改善 `10.05%`，可作为工程
>   Pareto组合，但不足以构成核心方法。详见
>   `reports/g2-asymmetric-stitch-gate-w0-20260821.md`。
>
> ## 2026-08-21 可转发执行摘要（Q0-A 设计与历史）
>
> ### 1. 当前核心方法
>
> 新候选方法暂称 **Dual-Expert Damage World Model（DE-DWM）**。它不是继续给
> DFWM 增加 latent/contact head，而是把已有正结果组合成不可旁路的 product-space
> 分工：
>
> - **Structural expert**：FT-GWM K1，保留锁定连杆的固定 SE(3) 几何，仅预测
>   joint state，并解析保证锁定关节位置/速度约束。
> - **Predictive expert**：ordinary constant-condition deep ensemble，负责
>   object state 与经验不确定性。
> - **Product-space fusion**：下一状态的 joint 来自 structural expert，object
>   来自 predictive expert；第一版冻结两个专家，不训练额外 gate。
> - **待验证核心量**：两个异构专家在 joint 子空间的分歧
>   `u_cross = RMSE(joint_data_expert, joint_structural_expert)`。目标是检测普通
>   ensemble 成员可能共同犯下、因而无法被内部 disagreement 暴露的结构错误。
>
> ### 2. Q0-A 融合保真结果
>
> - Q0-A 使用 leave-one-joint-out 冻结协议、相同训练/评估轨迹、seed 7/17。
>   主域 D3 mixed composition 的 object RMSE 分别从 `0.3103/0.1467` 变为
>   `0.3036/0.1438`（改善 `2.15%/1.98%`）；free-arm RMSE 分别改善
>   `53.86%/41.08%`；constraint violation 均为 `0`。两 seed 均通过 object
>   回退不超过 2%、free-arm 回退不超过 5%、violation 不超过 `1e-7` 的门槛。
> - 当前结论仅为 **Q0-A TWO-SEED PASS**：证明冻结异构专家能够组合且保持预测
>   保真。尚未证明 cross-expert discrepancy 提供独立风险信息，也未证明控制收益；
>   不得把 Q0-A 写成风险感知或控制性能已经成立。
> - 权威报告：`reports/g2-dual-expert-gate-q0a-20260821.md`。
>
> ### 3. 旧结论如何串联到新方法
>
> | 已有证据 | 对 DE-DWM 的约束 |
> |---|---|
> | ordinary ensemble 相对参数匹配单模型改善 `30.74%`，95% CI `[15.06%, 42.62%]`，5/5 seeds | 保留为 predictive expert |
> | structured vs ordinary ensemble 仅改善 `2.47%` 且 CI 跨零 | 不再把 topology conditioning 本身作为预测创新 |
> | 50% coverage 下 selective RMSE 下降约 `50.50%` | 保留 ensemble uncertainty，但固定深度重新校准 |
> | FT-GWM K0 PASS、K1 two-seed provisional PASS | 保留为 structural expert |
> | FT-GWM K2、FTC-WM L、hybrid-contact M、multi-contact N 均 No-Go | structural branch 不再学习 object/contact |
> | Guarded MPC 的统计区间跨零 | Q0-B 前不做控制收益主张 |
>
> 因此旧工作没有作废：它们构成了“预测专家擅长 object、结构专家擅长约束，任何
> 单一专家都不足”的证据链；但旧数字只能支持设计动机，不能替代 DE-DWM 的新实验。
>
> ### 4. MuJoCo Warp 加速试验
>
> 本机环境为 MuJoCo `3.11.0`、Warp `1.16.0`、RTX 4060 Laptop GPU。
> raw physics benchmark 结果为：32 worlds 时 CPU 约 `267k steps/s`、Warp 约
> `90k steps/s`；256 worlds 时 CPU 约 `177k steps/s`、Warp 约
> `678k steps/s`，Warp 约 `3.8x`。100 步一致性测试的 qpos/qvel RMSE 约为
> `8.3e-8/1.0e-7`。
>
> 结论：Warp 在数百环境的大 batch 下有价值，但当前每次约 24 条训练轨迹时反而
> 不能加速；首次 JIT 还需约 31 秒。因此现阶段保留 CPU MuJoCo 冻结数据协议，待
> Q0-B 需要数百/数千条校准轨迹时再接入 Warp。当前训练加速优先级是 FT-GWM 边
> 传播张量化、FK 缓存和 rollout 编译。
>
> ### 5. 下一步冻结决策
>
> Q0-B 的冻结计划为：
>
> 1. 在每个固定 rollout depth 分别计算 ensemble disagreement、`u_cross` 和真实误差；
> 2. 检验 `u_cross` 在控制 ensemble disagreement 后是否仍有独立解释力；
> 3. 比较 ensemble-only 与 ensemble + cross-expert risk score 的 selective AURC；
> 4. 只有 AURC 相对改善至少 `10%` 且至少 `4/5` seeds 方向一致，才进入 Q0-C
>    消融与 Guarded MPC。实际仅 3/5 通过，因此已执行停止规则。

> ## 2026-08-20 执行基线修订（优先于本文旧 G2 叙事）
>
> 本节同步 2026-08-20 后续实验结论。下文仍保留 DFWM-Hypernetwork 的 Seed 7
> smoke 记录，**仅作为已否定路线的审计历史，不得再被解释为待确认的正向结果**。
>
> ### 当前结果汇总
>
> | 路线 | 最终状态 | 已确认结果 | 不得主张 |
> |---|---|---|---|
> | 原始 DFWM latent/encoder/FiLM/dropout/hypernetwork | **NO-GO** | K-shot 独立贡献近零，跨 seed 不稳定 | latent adaptation 有效 |
> | CR-GWM / Gate E--H | **PROVISIONAL / attribution failed** | exact zero violation；Gate H 统一口径后 free-arm 仅退化 0.29% | reaction head 带来独立预测优势 |
> | RC-GWM / Gate I、J1--J6 | **NO-GO as stable model** | zero violation；数据多样性和优化协议已修复 | reduced-coordinate 模型跨 seed 稳定 |
> | FT-GWM / K0 | **PASS** | 固定 SE(3) 链与完整链、MuJoCo 位姿机器精度一致 | 已证明动力学优势 |
> | FT-GWM / K1 | **TWO-SEED PROVISIONAL PASS** | D3 seed 7/17 free-arm 相对变化 `+3.45%/-28.81%`，violation=0 | 统计稳定或 compute-matched 优势 |
> | FT-GWM / K2 | **NO-GO** | stop-gradient 严格隔离 object loss；K1 joint fidelity 被保留 | 完整 Push object/contact 预测成功 |
> | Ensemble uncertainty / selective prediction | **当前主线** | 五 seed 证据；50% coverage 下 RMSE 约降低 51% | 未经验证的稳定控制收益 |
>
> - 原始 DFWM 的 residual latent、amortized encoder、FiLM/residual adapter、topology
>   dropout 与 hypernetwork 分支均为 **No-Go**：五 seed 审计中 K-shot 的独立贡献近零，
>   `z` 范数约 0.07--0.09，跨 seed 不稳定。后续不再投入该路线，也不以 DFWM 命名主方法。
> - shared chain graph dynamics 显著优于旧 dense GRU；但 matched graph 消融表明，这一增益
>   主要来自图架构本身，不能归因于 topology conditioning。
> - 当前唯一保留的机制候选为 **Constraint-Reaction Graph World Model (CR-GWM)**：冻结共享链图
>   base，依据已诊断 joint lock 的预测约束残差沿运动链传播 reaction，只修正自由关节和物体，
>   对锁定关节的位置和速度实施解析投影。
> - Gate E（D3 完全 held-out；训练仅 intact+D2+D4；`D3__mixed_composition`；5 seeds）相对
>   graph ordinary 的改进：object RMSE **+41.33%**, 95% CI **[+20.09%, +59.79%]**；free-arm
>   **+5.54%**, **[+0.84%, +10.07%]**；overall **+15.87%**, **[+11.97%, +19.63%]**；所有
>   评估域的锁定位置/速度 violation 为 **0**。结论为 **PROVISIONAL PASS**，详见
>   `reports/g2-constraint-reaction-gate-e-20260820.md`。
> - Gate F（seed 7 公平性审计）尚未通过方法归因：parameter-matched graph（299,782 参数）优于
>   CR-GWM（291,373 参数）的 overall/free-arm 指标；同容量 unconstrained residual adapter 的
>   object 指标也优于 CR-GWM。CR-GWM 目前唯一经确认的专属优势是 exact zero constraint
>   violation。因此不得声称其预测优势超过同容量基线。
> - Gate G 的原始否定记录为：direct lock projection 从 overall/free-arm/object
>   `0.1712/0.2016/0.0306` 变为 `0.2281/0.2978/0.0691`。后续审计发现该 projection
>   模型误用了 `hidden=96`，并非 matched graph；此记录仅保留为审计历史，其结论已由下方
>   Gate H 修正版结果撤回。
> - **下一步唯一实验：Gate H（仅 seed 7）**。实现 `hidden=128` matched graph + 低容量 gated
>   reaction head（gate 近零初始化）+ exact projection，并与 matched graph、direct projection、
>   unconstrained residual adapter 比较。仅当 violation 约为 0，且 object 与 free-arm 相对
>   matched graph 的退化均不超过 5%，才扩展至五 seed；否则停止 CR-GWM 主线并重定位为
>   constraint-satisfaction benchmark/负结果。
>
> ### 2026-08-20 Gate H 最终执行结论（取代上条“下一步”状态）
>
> - Gate H seed 7 已按冻结配置完成。`hidden=128` matched graph 的
>   overall/free-arm/object RMSE 为 `0.1712/0.2016/0.0306`；低容量 gated reaction head
>   （2,744 个可训练参数）+ exact projection 为 `0.1612/0.2127/0.0220`，constraint
>   violation 为 `0`。相对 matched graph，object 改善 `27.93%`，但 free-arm 退化
>   `5.51%`，超过预注册上限 `5%`，因此判定 **NO-GO**，不扩展至五 seed，并停止
>   CR-GWM 主线。
> - 旧 Gate G 的 direct-projection 否定结论存在容量配置错误：runner 对
>   `graph_matched_projected` 使用了 `hidden=96`（169,542 参数），而参考模型为
>   `hidden=128`（299,782 参数）。修正后 matched direct projection 为
>   `0.1614/0.2123/0.0321` 且 violation 为 `0`；旧 Gate G 的“大幅损害预测”结论撤回，
>   但修正版 free-arm 仍退化约 `5.29%`，也未通过 5% 保真阈值。
> - 后续主线转为已有五 seed 强证据的 ensemble uncertainty / selective prediction；
>   Gate E--H 仅保留为 constraint-satisfaction benchmark 与负结果链。完整审计见
>   `reports/g2-gated-reaction-gate-h-20260820.md`。
>
> ### 2026-08-20 指标审计更正与 Gate I
>
> - Gate H 的 `+5.51%` free-arm 退化来自不一致口径：matched graph 错误地按全关节统计，
>   gated reaction 按真实自由关节统计。统一使用真实 damage mask 后，matched graph free-arm
>   RMSE 为 `0.2121`，gated reaction 为 `0.2127`，退化仅 `0.29%`。因此撤回 Gate H No-Go，
>   更正为 **PROVISIONAL PASS**；不扩展该 head，因为 Gate I 提供了更简洁的内生约束方案。
> - Gate I 的 RC-GWM 在动力学图中移除锁定坐标、跨锁定节点重连最近自由关节、屏蔽锁定节点
>   recurrent state，并仅用自由节点预测 object。seed 7 primary D3 的 matched graph / RC-GWM
>   overall/free-arm/object 为 `0.1712/0.2121/0.0306` 与 `0.1586/0.2095/0.0153`；RC-GWM
>   violation 为 `0`，object 改善 `50.00%`，free-arm 改善 `1.22%`，Gate I **PASS**。
> - 下一步冻结为 Gate I 五 seed 扩展；`D3__mixed_unseen` 必须单列为 failure boundary，不得被
>   primary composition 的正结果掩盖。详见 `reports/g2-reduced-coordinate-gate-i-20260820.md`。
>
> ### 2026-08-20 Gate I 五 seed 结论
>
> - RC-GWM 五 seed primary `D3__mixed_composition` 仅 `2/5` 通过：seed 7/47 通过，
>   seed 17/27/37 失败。所有 seed violation 均为 `0`，但失败 seed 的 free-arm 退化为
>   `37.28%--84.00%`，说明坐标约简的可行性成功而自由臂预测稳定性失败。
> - RC-GWM 具有比 topology token、direct projection 和 reaction adapter 更清晰的结构创新，
>   但当前实现不得作为稳定主方法；不再进行未预注册调参。完整审计见
>   `reports/g2-reduced-coordinate-gate-i-5seed-20260820.md`。
> - 后续若要复活该方向，必须先提出新的稳定性机制（例如 free-arm/object 解耦 head 与梯度
>   冲突控制）并重新冻结实验；当前论文主线回到已有五 seed ensemble uncertainty /
>   selective prediction 证据。
>
> ### 2026-08-20 J6 数据协议修复后的最终诊断
>
> - `goal_exploration_std=0.08` 的低通有界探索使不同 seed 训练轨迹真正不同，且接触/方块位移
>   与旧协议一致；`lr=1e-3, 60 epochs` 消除了 seed 17 的 catastrophic rollout 发散。
> - 但 RC-GWM seed 7/17 primary free-arm 仍为 `0.2436/0.2448`，相对 matched graph 约退化
>   `15%`，而 object 约为 `0.0086` 且 violation 为 `0`。因此数据与优化问题已修复，剩余问题
>   是 reduced-coordinate 归纳偏置损害自由臂动力学；RC-GWM 不作为稳定主方法继续扩展。
> - 不再进行 generic edge feature、packed slot 或未注册 loss 权重堆叠。若未来复活，必须使用
>   保留锁定连杆完整物理变换的 free-joint dynamics 架构；当前论文主线保持 ensemble uncertainty /
>   selective prediction。
>
> ### 2026-08-20 RC-GWM 逐原因诊断结论
>
> - J1/J1b 确认主要原因是 rollout 优化多稳态：seed 17 将学习率从 `3e-3` 降至 `1e-3`
>   并以 60 epochs 匹配累计预算后，primary free/object 从 `0.3882/0.0645` 改善到
>   `0.2428/0.0086`，但 free-arm 仍差于 matched graph 的 `0.2110`。
> - J2 确认 object 与 joint graph 的共享梯度/递归耦合是贡献因素，但独立 stop-gradient 仍不能
>   恢复 seed 17。J3 的普通 bridge edge 特征无效；J4 的真正 packed active-node graph 与 masked
>   实现逐数值相同，证明两者在共享 permutation-equivariant 模型下等价。
> - J5a 数据审计发现 `goal` 采集不使用随机 seed：seed 7/17 的训练轨迹逐元素相同；增加
>   trajectories 只循环有限 targets，不增加独立信息。因此五 seed 主要审计初始化稳定性，而非
>   数据抽样稳定性。
> - J3b 说明收缩锁定节点时丢失固定关节变换是物理建模缺陷，但加入 lock angle sin/cos 仍未
>   修复 free-arm。未来必须使用 URDF-derived SE(3) transform composition，而非继续堆通用 edge
>   feature。完整报告见 `reports/g2-rcgwm-root-cause-diagnosis-20260820.md`。


> ### 2026-08-20 Gate K0/K1 固定变换图结论
>
> - RC-GWM 的核心物理缺陷已被修正：关节锁定后连杆不会消失，而是形成固定 SE(3) 变换。K0 在 D2/D3/D4、每种 100 个随机姿态上与完整链和 MuJoCo 末端位姿达到机器精度一致，判定 **PASS**。
> - FT-GWM 保留五个链节点，把锁定节点作为固定几何和消息中继，仅预测自由关节。冻结协议为 hidden=128、lr=1e-3、60 epochs、探索噪声 0.08，训练 intact+D2+D4，主评估为 held-out D3。
> - K1 seed 7：matched graph / FT-GWM free-arm RMSE 为 `0.2611/0.2701`，相对退化 `+3.45%`；seed 17 为 `0.3891/0.2770`，相对改善 `28.81%`。两 seed 所有域 constraint violation 均为 `0`，均通过预注册的退化不超过 5% 门槛。
> - FT-GWM 参数量 `267,650`，低于 matched graph 的 `299,782`，但显式逐边 SE(3) 实现训练更慢，尚未 compute-match。当前结论为 **K1 two-seed provisional PASS**：证明固定变换表示可行，不声称统计稳定的预测优势。
> - K1 通过后按冻结规则执行 K2；其最终 No-Go 结果见下节。详见 `reports/g2-fixed-transform-graph-gate-k1-20260820.md`。
>
> ### 2026-08-20 Gate K2 最终结论
>
> - K2 增加 2,340 参数的 bottleneck-16 object residual head，输入当前 object state、detach 后的 joint hidden 与末端 SE(3)；自动梯度测试确认纯 object loss 对 joint transition 的梯度逐元素为零。
> - K2 v1 因把 joint/object 维度统一平均而将 K1 joint 梯度缩小为 `10/14`，协议无效。v2 修正为 `L_joint + L_object`，保持 K1 的 joint loss 尺度。
> - v2 seed 7 primary D3：matched graph overall/free/object 为 `0.1716/0.2212/0.0104`，FT-GWM K2 为 `0.2130/0.2701/0.1133`，violation 为 `0`。FT free-arm 与 K1 的 `0.2701` 完全一致，说明隔离成功；但相对 object-aware matched graph，free-arm 退化 `22.11%`、object 退化 `986.08%`。
> - Gate K2 判定 **NO-GO**，按预注册规则停止 FT-GWM 完整 Push world-model 分支，不追加容量、接触特征、loss 权重或 epochs。K1 只保留为 constraint-preserving joint-dynamics 正结果；稳定主线仍为 ensemble uncertainty / selective prediction。
>
> ### 2026-08-21 FTC-WM Gate L 最终结论
>
> - Gate L 将 contact/free-object 分支显式隔离并保留 pusher 几何。模型稳定收敛，60 轮 loss
>   从 `0.3990` 降至 `0.0371`，但仍为 matched baseline `0.0176` 的约 `2.11x`；20--40 轮
>   未进入 baseline 区间。
> - 四个评估域的 object rollout RMSE 为 `0.2209--0.2615`，平均约 `0.247`；K2 v2
>   平均约 `0.103`，Gate L 反而恶化约 `2.4x`。汇总回归为 free-arm `18.15%`、object
>   `885.63%`，`gate_passed=false`。
> - Gate L 判定 **NO-GO**。该结果说明显式 contact/free-object 分支在冻结预算内仍未解决
>   Push object dynamics；不追加 epoch、容量或 loss 权重，作为 K2 后续反证归档。

---

## 0. 一页执行摘要

### 0.1 核心决策

旧路线“random mask + morphology token + actor-head fine-tuning”存在三个无法靠补实验解决的问题：

1. 训练覆盖测试 mask，不能支持“未见离散损坏适应”；
2. “连续 embedding 不能表示离散变化”不成立；
3. token 与 actor 同时更新，无法判断恢复来自哪个组件。

V6 根据 G1 的反证结果再次收缩主张。原始 residual latent、history/FiLM
adapter 和 Reach 优势均未形成稳定证据，旧 Push 15.8% 结果因零接触协议失效。
当前主线改为：

> **Robust Zero-Shot Structured Dynamics**：利用诊断可得的离散损坏拓扑训练多个独立条件动力学模型，在未知故障强度和 held-out 组合上通过集成均值提高多步预测，并以模型分歧提供经验不确定性。部署时冻结模型，不依赖 residual latent 在线适配。

DFWM 保留为被否定的原始假设和对照方法，不再把“少量试运行识别故障严重度”
作为当前已成立贡献。Guarded MPC 仅为次要控制验证，除非 G2 统计区间不跨零，
不得升级为稳定控制收益主张。

### 0.2 ICRA 2027 投稿约束与会议策略

| 目标 | 官方状态（2026-08-06 核验） | 项目策略 |
|---|---|---|
| **ICRA 2027** | 常规论文截止为 **2026-09-15 11:59 PM PST**；完整论文（正文、图表、表格、致谢、参考文献）总计最多 8 页；双栏、双匿名、PDF 投稿。配套视频最多 180 秒、20 MB，首个上传窗口为 2026-08-05 至 09-09，第二窗口为 09-17 至 09-22。[官方投稿说明](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) | **唯一主目标**；所有资源、证据和排期围绕该截止倒排 |
| **RSS / CoRL 2027** | 当前不参与本轮资源排期 | 仅作为 ICRA 未提交、被拒或主动撤退后的后续去向，不得稀释当前执行目标 |

ICRA 提交还必须满足：至少选择 3 个官方关键词；PaperPlaza 元数据与匿名稿一致；正文不得包含可识别作者身份的信息；除可选视频外，不假设审稿人会访问外部链接或补充材料；投稿前再次核验 PDF compliance 与 IEEE-RAS AI 使用披露要求。

### 0.3 总体路线

| 阶段 | 目标 | 主要成本 | 关键决策 |
|---|---|---:|---|
| G0 | 5-DoF+夹爪运动学、可达性、硬件与物理测量 | 16–24 工时；4–8 真机小时 | URDF 与真机是否一致、任务是否物理可做 |
| G1 | 最小机制验证 | 40–60 工时；30–60 GPU-h | 原始假设 No-Go；robust zero-shot Pivot-Go |
| G2 | ICRA 核心仿真证据 | 60–90 工时；60–100 GPU-h | 是否形成稳定方法贡献 |
| G3 | 真机重复验证 | 30–45 工时；8–16 真机小时；8–16 GPU-h | 是否支持真实恢复主张 |
| G4 | 论文、视频与投稿 | 30–45 工时 | 是否达到投稿完整性 |

**ICRA 冲刺预算**：约 176–264 工时、98–186 GPU-h、12–28 真机小时、75–240 GB 存储。
**成本控制原则**：G1 不通过，不进入 G2；G2 不通过，不投入正式真机统计。

---

## 当前执行状态（2026-08-20 更新）

### G0

G0 已完成并通过，交付物、真机校准、MuJoCo 模型、可达域、锁定安全、急停和 10 姿态 TCP 记录均已归档。G0 仍保留“后 5 个姿态为用户确认一致而非独立尺量”的证据说明。

### G1

原始 DFWM residual 路线判定 **No-Go**。Reach 的早期优势未通过多 seed
复核；旧 Push 15.8% 结果使用了零接触、零方块位移的错误协议，禁止作为论文
证据。residual latent、history encoder、静态/动态 FiLM 和 residual correction
均未通过 fidelity-stable D2/D3 gate。

Push 协议已修正：补齐夹爪下指碰撞几何、分离 IK waypoint 与任务目标、冻结
不重叠 target split，并要求 D2/D3 评估轨迹存在真实接触和方块位移。在该协议上，
三成员 topology-conditioned ensemble 相对参数匹配单模型的五 seed 平均多步
RMSE 改善为 **30.7%**，seed bootstrap 95% CI 为 **[15.1%, 42.6%]**；
D2/D3 和 5/5 seeds 方向均为正。参数量分别为 450,906 与 460,382。

Guarded MPC 在五目标审计中改善 11/15 个 seed-target 组合，15/15 保持任务
成功，但三 seed 区间跨零。因此当前状态为：**G1 原始方法 No-Go；G1 robust
zero-shot Pivot 通过最小预测机制门并可阶段交付；G2 可启动，但控制收益仍是
次要、未证实结论。**权威结果见
`reports/g1-robust-zero-shot-corrected-results-20260819.md` 和
`results/final/g1_robust_zero_shot_5seed_summary.json`。

### G2

**2026-08-19 首轮强基线**：structured vs ordinary ensemble，5 seeds。平均改善
**2.47%**，95% CI **[-1.83%, 6.38%]**，触发 Pivot。

**2026-08-19 诊断实验**：GRU hidden-state 线性探针。结论：topology descriptor
在当前设定下提供冗余信息（conditioning redundancy，非 collapse）。

**2026-08-19 held-out topology 实验**：D3 held-out 平均改善 **+0.02%**，CI 跨零。

**2026-08-19~20 DFWM 落地尝试（共 8 种方法，均失败）**：
- Latent optimization、Amortized encoder（多版本）、两阶段训练、物理监督+对比学习、
  Topology Dropout 在 in-distribution 场景均导致 z_norm≈0.2（posterior collapse）
- Oracle（真实物理参数作为 z）比 ordinary 还差，确认 WM 架构级别忽略 z
- 分歧指纹识别：K=1 → **100% 识别 D2/D3**（5/5 seeds），但识别后预测不改善
- 根本原因：concat context 允许 WM 忽略 z；需要架构级别变更

**2026-08-20 超网络架构（DFWM-Hypernetwork，Seed 7；已归档为 No-Go）**：
- OOD split：训练只见 nominal+weak_motor，测试遇到 high_damping+delay_1
- 架构：`z → HyperNet(LoRA) → ΔW`，修正量 = `hidden @ (W_base + ΔW) + bias(z)`
- 两阶段训练（Stage1 WM_base，Stage2 冻结 WM 只训练 HyperNet+encoder）
- **Seed 7 结果**：D2 high_damping +6.5%，D2 delay_1 +4.5%，平均 **+4.3%**
- Oracle 比 base 好（oracle_imp 正值）——WM 首次学会利用 z 信息
- **待验证**：K=0 vs K=5 差异微小（0.1%），K-shot 贡献尚未独立确认

**历史状态（已被文首 2026-08-20 执行基线修订取代）：超网络曾有单 seed 信号；五 seed
复核后确认 K-shot/latent 机制不成立。**

**关键未决问题**：
1. 5 seeds 超网络结果是否稳定（CI 是否不跨零）
2. 去掉 W_base 静态残差通道后，K-shot 是否有 >2% 独立贡献
3. 如两问均确认：DFWM 方法论文成立；否则转架构贡献或 benchmark 定位

已完成交付物：
- `config/experiment/g2_push_ensemble_v1.yaml`（冻结协议）
- `config/experiment/g2_push_heldout_topology_v1.yaml`
- `config/experiment/g2_dfwm_ood_v1.yaml`（超网络 OOD 实验配置）
- `config/splits/g2_dfwm_ood_v1.yaml`（OOD split 定义）
- `results/final/g2_structured_vs_ordinary_5seed.{json,csv}`
- `results/final/g2_heldout_topology_5seed.{json,csv}`
- `results/final/route2_topo_id_5seed.{json,csv}`（分歧指纹识别结果）
- `results/final/route2_structured_topo_id_5seed.{json,csv}`
- `runs/g2_push_ensemble/` 5 seeds
- `runs/g2_heldout_topology/` 5 seeds
- `runs/g2_domain_randomized/` 5 seeds
- `runs/g2_dfwm_hypernetwork/seed7_v1/`（超网络 smoke test）
- `scripts/run_g2_dfwm_hypernetwork.py`（超网络训练+评估脚本）
- `scripts/collect_warp.py`（MuJoCo Warp GPU 批量采集）
- `src/robotarm/models/amortized_encoder.py`（ResidualEncoder + 物理监督）
- `reports/g2-ordinary-ensemble-gate-20260819.md`
- `reports/g2-heldout-topology-gate-20260819.md`
- `reports/route2-topo-id-gate-20260820.md`
- `HANDOFF-2026-08-20.md`（Codex 接力文档）

## 1. 项目目标与成功定义

### 1.1 科学目标

研究低成本串联机械臂发生单关节锁定后，已知故障拓扑的结构化条件动力学
集成能否在未知故障强度下提供更稳健的多步预测、可用的不确定性排序，并最终
支持安全控制。少样本 residual identification 降为已受反证的备选问题。

### 1.2 工程目标

交付一个可复现系统，包含：

- 5-DoF 机械臂加独立夹爪 MuJoCo 模型；
- URDF—舵机—真机坐标映射和经过实测校准的运动链；
- 可配置的关节锁定、摩擦、顺应性、背隙和延迟模型；
- 仿真与真机统一轨迹接口；
- topology-conditioned dynamics ensemble；
- parameter/compute-matched baselines 与 ensemble disagreement；
- 至少 Reach 和 Push 两个任务；
- 可重跑的实验配置、日志、checkpoint 和统计脚本；
- 真机校准协议、视频与安全记录。

### 1.3 项目级成功条件

项目达到“机器人顶会可投稿”必须同时满足：

1. **机制成立**：冻结模型时，structured ensemble 相对普通 deep ensemble 和参数/算力匹配单模型仍有稳定收益；
2. **非记忆**：在训练未出现的 topology–physics 组合上仍有效；
3. **基线可信**：至少覆盖 topology-only single、ordinary deep ensemble、domain-randomized ensemble 和 parameter/compute-matched single；
4. **真实重复**：真机不是单段展示；最低 ICRA 证据包覆盖两个故障条件且每条件不少于 20 个 evaluation episodes，强证据包每条件不少于 30 个；
5. **成本透明**：报告交互步数、真实秒数、适配时间、GPU-h、失败次数；
6. **可复现**：从环境创建到表格生成存在单命令或明确脚本链；
7. **主张克制**：所有结论与结果严格对应，不使用未验证成功率或“first”式主张。

---

## 2. 研究问题、假设与非主张

### 2.1 问题设定

给定一个已训练的机械臂控制系统。部署时发生单关节锁定：

- 故障诊断模块或人工检查能够提供锁定关节身份；
- 锁定角可以直接读取或粗略测量；
- 摩擦、顺应性、回差、负载和延迟等 residual dynamics 未知；
- 主设置为 zero-shot，不使用评估实例的在线校准轨迹更新模型；
- 训练、验证和评估目标及 physics composition 严格分离。

### 2.2 主要研究问题

- **RQ1**：已知故障 topology 的结构化条件是否能提高 held-out physics composition 的多步预测？
- **RQ2**：收益是否超过普通 deep ensemble，并在参数量和训练 compute 对齐后保留？
- **RQ3**：ensemble disagreement 能否在 rollout depth 分层后稳定排序预测误差？
- **RQ4**：预测改善是否能转化为冻结控制器的稳定控制收益？
- **RQ5**：该方法在哪些故障强度、接触条件和目标区域失效？

### 2.3 可证伪假设

- **H1 Structured prediction**：在 held-out topology–physics 组合上，structured ensemble 的多步误差低于普通 deep ensemble。
- **H2 Fairness**：在总参数量和训练 compute 分别对齐后，H1 的方向与区间仍成立。
- **H3 Uncertainty**：depth-stratified ensemble disagreement 与多步误差正相关；若不成立，不主张校准不确定性。
- **H4 Control transfer**：冻结 guarded planner 的控制改善在五 seed、多目标下区间不跨零；若不成立，控制只作负结果。
- **H5 Boundary**：优势在接触丰富、残余物理可影响状态转移的 Push 中强于简单 Reach，并存在可解释 failure regime。

### 2.4 明确不再主张

- 不主张连续向量无法表达离散故障；
- 不主张随机 mask 本身是新算法；
- 不把训练中出现过的 joint mask 称为 unseen morphology；
- 不把 actor-head fine-tuning 的收益归因给 morphology token；
- 不把 intact robot 表现当作 damaged morphology 的唯一 oracle；
- 不在真实数据产生前承诺 60%、80% 或固定胜幅；
- 不把“低成本平台”本身当作算法新颖性。
- 不再使用旧 Push 15.8% 数字；
- 不声称 residual latent 已识别故障严重度；
- 不把集成平均本身包装成结构化方法创新；
- 不在控制置信区间跨零时声称稳定恢复提升。

---

## 3. 研究范围与降维策略

### 3.1 主线范围

| 维度 | 主线 |
|---|---|
| 平台 | 现有 5-DoF GenkiArm 3D 打印臂 + 独立夹爪 + Feetech STS3215 |
| 故障 | 单关节锁定 D2 / D3 / D4 |
| 任务 | Reach、Push |
| 观测 | proprioception + 任务状态；RGB 仅用于目标/物体定位与视频 |
| 动作 | 5 维关节位置/增量命令；锁定关节动作由 adapter 屏蔽；夹爪开合作为独立执行器，不计入 5-DoF 定位链 |
| 仿真 | MuJoCo |
| 真机 | Feetech SDK + Python + websockets；不要求 ROS |
| 模型 | topology-conditioned dynamics ensemble；conditional RSSM 作为成员实现 |
| 部署更新 | zero-shot；world model 与 planner 冻结 |

### 3.1.1 真实机械链与命名冻结

URDF `genkiarm.urdf` 给出的串联链为：

```text
Base --J1--> Yao --J2--> Jian1 --J3--> Jian2
     --J4--> Wan --J5--> Wan1 --J6--> Zhua
```

| 统一编号 | URDF joint | 功能角色 | URDF 轴 | 名义 origin xyz (m) | 主实验状态 |
|---|---|---|---|---|---|
| J1 | `Rotation` | 底座旋转 | X（joint frame） | `[-0.013, 0, 0.0265]` | 完整建模；不作为首批锁定故障 |
| J2 | `Rotation1` | 中间关节 1 | Y | `[0.081, 0, 0]` | D2 主故障 |
| J3 | `Rotation2` | 中间关节 2 | Y | `[0, 0, 0.118]` | D3 主故障 |
| J4 | `Rotation3` | 中间关节 3 | Y | `[0, 0, 0.118]` | D4 主故障 |
| J5 | `Rotation4` | 手腕 | Z | `[0, 0, 0.0635]` | 完整建模；姿态故障扩展 |
| J6 | `Rotation5` | 夹爪整体姿态 | X | `[0, -0.0132, 0.021]` | 完整建模；姿态故障扩展 |

“夹爪自由度”在本文中指 J6 对夹爪整体姿态的控制。若实体夹爪还有手指开合电机，则记录为 `gripper_open` 独立执行器，不将其误计为第七个机械臂定位自由度。URDF 中所有 ±1.57 rad 仅为名义限位，真实软限位必须经 G0 测量后覆盖。

### 3.2 延后项目

以下内容只有在 G2 通过后才允许加入：

- Pick/Place；
- RGB end-to-end world model；
- 双关节损坏；
- 未知故障身份的在线诊断；
- uncertainty-aware active probing；
- 多机械臂或多 embodiment；
- GNN morphology encoder；
- LoRA 大矩阵；
- ROS/MoveIt 集成。

### 3.3 删除或替换

- `From-scratch-5 RL` 替换为 `BC-from-5` 或 `offline learner-from-5`；
- “Danesh exact reproduction”替换为 matched continuous-descriptor baseline；
- 8-baseline × 9-cell 全量矩阵缩减为 4 个因果 baseline × 6 个主 cell；
- Pick 仅在锁定后可达性达到门槛时进入附加实验。

---

## 4. 技术方案

### 4.0 V6 主方法冻结

V6 主方法由三个共享训练协议但独立初始化的 topology-conditioned dynamics
members 组成。推理采用 ensemble mean；不确定性采用成员预测分歧。所有成员在
部署时冻结，不使用 test-instance residual optimization。参数匹配宽单模型、普通
deep ensemble 和 domain-randomized ensemble 必须共享数据、训练轮数、优化器和
评估轨迹。以下 residual-context 小节保留为原始 DFWM 设计与失败 baseline 说明，
不再代表 V6 主方法。

### 4.1 原始 DFWM 损坏上下文（历史 baseline）

定义：

```text
c_damage = [e_topology(m, q_lock, joint_attributes), z_residual(D_K)]
```

其中：

- `m ∈ {0,1}^6`：六个定位关节的可用性；
- `q_lock`：锁定关节角度；
- `joint_attributes`：关节轴、真实范围、父子拓扑、链深度及 `base/intermediate/wrist/gripper-orientation` 功能角色；
- `D_K`：K 条校准轨迹；
- `z_residual ∈ R^d`：未知物理残差的低维表示。

`d` 的初始候选为 4、8、16；G1 默认使用 8，最终由 validation 而非 test 选择。

### 4.2 Topology encoder

禁止只使用“每个关节一个独立 lookup embedding”作为最终方案，因为它难以支持组合解释。推荐：

1. 每关节构建 `[presence, lock_angle, axis, normalized_limits, depth]`；
2. 共享 MLP 编码每个关节；
3. 按机械链顺序拼接，或用轻量 attention/pooling；
4. 输出固定维度 `e_topology`。

G1 可先实现按顺序拼接的共享 MLP；GNN/Transformer 不是前置条件。

### 4.3 Residual context 推断（历史 baseline，G1 No-Go）

按工程风险从低到高实施：

#### A. Latent optimization（G1 默认）

- 为每个部署实例初始化 `z_residual = 0`；
- 冻结 WM 和 actor；
- 最小化校准轨迹上的 multi-step prediction loss；
- 只更新 `z_residual`；
- 优点：实现简单，因果归因清楚；
- 缺点：每次部署需要梯度步骤。

#### B. Amortized encoder（G2）

- 输入最近的 `(o_t, a_t, o_{t+1})` 序列；
- 输出 residual posterior 的均值和方差；
- 可用 latent-optimization 结果作为训练 target 或直接端到端训练；
- 优点：推断快；
- 缺点：实现和训练成本更高。

#### C. Active calibration（可选 G2+）

- 从安全动作库中选最大化预测分歧或 posterior information gain 的动作；
- 必须满足软限位、速度、温度/电流和工作区约束；
- passive calibration 无效时不得直接上此模块。

### 4.4 World model

世界模型预测：

```text
p(o_{t+1}, r_t, continue_t | o_t, a_t, e_topology, z_residual)
```

最低实现要求：

- deterministic recurrent state；
- stochastic latent；
- observation、reward、continue heads；
- topology/residual 在 recurrent transition 与 prediction heads 均可访问；
- 训练期间记录 one-step 与 multi-step error；
- 支持 actor-free 的 rollout prediction smoke test。

### 4.5 Policy

首选两种实现路径：

1. **Dreamer-compatible actor-critic**：与 world model latent imagination 一致；
2. **MPC/short-horizon planner**：若 actor 训练不稳定，可用于验证 world model 本身。

G1 必须至少有一个冻结部署策略。若只有 actor fine-tuning 后能恢复，则项目自动触发 Pivot B。

### 4.6 训练分布

训练 domain 不使用未经测量的单点物理常数。G0 后根据真机测量确定：

- lock angle 范围；
- locked-joint static/dynamic friction；
- compliance 或等效弹簧参数；
- backlash；
- command latency；
- payload；
- servo tracking noise。

每个参数使用分层区间。训练/验证/测试按组合切分：

- 每个单独参数水平在训练中出现；
- 某些 `joint × lock angle × residual bin` 组合只在测试出现；
- test 不是简单从同一分布随机抽样；
- 切分写入不可变 YAML 并记录哈希。

### 4.7 真机与仿真 adapter

统一接口：

```python
class RobotEnv(Protocol):
    def reset(self, *, target, damage_config) -> Observation: ...
    def step(self, action) -> StepResult: ...
    def emergency_stop(self) -> None: ...
    def close(self) -> None: ...
```

仿真实现 `MujocoArmEnv`；真机实现 `FeetechArmEnv`。训练代码不得直接依赖 Feetech SDK 或 MuJoCo API。

---

## 5. 任务与故障协议

### 5.1 Reach

- 目标：末端到达 3D 目标位置；
- 主指标：成功率、最终距离、到达时间；
- 初始成功阈值沿用 5 cm 仅作工程起点，G0 根据相机和运动学误差校准；
- calibration targets 与 evaluation targets 不重合；
- target 只从健康与损坏 morphology 的共同可达域采样；
- 主指标采用 position-only Reach，末端姿态作为次指标，避免把 J5/J6 的姿态能力与 J2–J4 的位置可达能力混为一谈。

### 5.2 Push

- 目标：推动方块进入目标区域；
- 主指标：成功率、最终物体距离、碰撞/越界；
- 必须先固定物体尺寸、摩擦面和相机标定；
- 评估分 easy/medium/hard target bins，但主结论预先选定一个 aggregate。
- Push 除位置误差外记录夹爪/末端接触姿态；若 J5/J6 未被控制稳定，不进入正式主表。

### 5.3 Pick（条件性）

进入条件：

- 锁定后 position-only 可达率足够；
- 抓取器与物体检测已稳定；
- 健康策略在仿真和真机均达到可重复基线；
- 不影响主线日期。

未通过则删除，不将不可达任务失败解释为适应失败。

### 5.4 故障条件

- D1：J1 底座锁定，主要压缩方位工作区，仅作扩展；
- D2：J2 中间关节 1 锁定，主故障；
- D3：J3 中间关节 2 锁定，主故障；
- D4：J4 中间关节 3 锁定，主故障；
- D5：J5 手腕锁定，主要影响末端姿态，仅作扩展；
- D6：J6 夹爪整体姿态锁定，仅作扩展；
- 每种故障至少包含多个 lock angles；
- “软件把动作设为 0”与“高摩擦物理锁定”必须分开；
- 真机 screw fixation 的实际微动由测量决定，不沿用 V3 的假设值。

ICRA 主实验固定为 D2/D3/D4，因为它们直接改变位置可达性、冗余与动力学，且三者具有相同的中间关节功能族，便于公平比较。D1 与 D5/D6 的故障后任务定义不同，不与主结果平均；只有主证据包完成后才作为边界分析。

---

## 6. 实验与证据矩阵

### 6.1 核心方法

- DFWM + passive residual calibration；
- 可选 DFWM + amortized residual encoder；
- 可选 DFWM + active calibration。

### 6.2 必需 baseline

| Baseline | 回答的问题 | 公平性 |
|---|---|---|
| Topology-only zero-shot | 真实轨迹是否必要 | 同一 WM、同一 topology、`z=0` |
| History encoder | RMA/OEA 类短历史推断是否已足够 | 同预训练数据、同观测历史、近似参数量 |
| Matched continuous descriptor | factorization 是否比单一连续 descriptor 有效 | 同网络容量、同真实数据预算 |
| Parameter-matched adaptation | 收益是否只是多了可训练参数 | 同 trainable params、同更新步数 |

### 6.3 条件 baseline

- Full fine-tune：仅当 G1 表明小 context 有优势时加入；
- BC-from-5：用于识别是否只是 few-shot imitation；
- Damaged oracle：获得真实 residual 参数或充分损坏数据；
- Intact oracle：只作为健康参考，不作为损坏上限；
- LoRA：仅当 reviewer 风险仍指向 generic adapter 时加入。

### 6.4 必需消融

1. topology-only；
2. residual-only；
3. factorized；
4. actor frozen vs actor-head updated；
5. latent dimension 4/8/16；
6. K = 0/1/2/5/10；
7. random combination split vs held-out combination split；
8. latent optimization vs amortized encoder（G2）。

### 6.5 主要指标

定义 normalized recovery：

```text
NR = (S_adapted - S_no_adapt) / max(S_damaged_oracle - S_no_adapt, ε)
```

其中 `S` 为同一损坏条件下的 success 或 return。必须同时报告原始 success，避免比例掩盖。

其他指标：

- success rate；
- mean return；
- final position/object error；
- calibration transitions；
- calibration wall-clock；
- context optimization wall-clock；
- WM one-step / multi-step NLL；
- unsafe action / emergency-stop count；
- GPU-h；
- peak memory；
- trainable parameter count。

### 6.6 统计方案

- G1：3 training seeds，用于方向与机制闸门，不做强显著性主张；
- G2：5 training seeds；
- 仿真每个 seed/condition 至少 50 evaluation episodes；
- 真机最低包每 condition 至少 20 episodes；强包至少 30 episodes，并跨至少 3 个 target sets 或 3 个实验日；
- 使用 hierarchical paired bootstrap：先重采样 seed，再重采样 seed 内 targets；
- 主要比较预先限定为 3 个，必要时使用 Holm correction；
- 报告效应量与 95% CI；
- 不能把同一训练模型的多次 rollout 当作独立训练重复；
- 不用 5 seeds 的双侧 Wilcoxon 星号支撑核心结论。

---

## 7. 阶段门与 Definition of Done

## G0 — 物理与可达性基线

**建议日期**：2026-08-06 至 2026-08-11
**负责人类型**：项目本人；必要时机械/控制同学复核  
**后续 owner**：`ccf-experiment-designer` 负责将测量结果固化为最终实验协议

### 输入

- 实体机械臂；
- 现有网页控制接口；
- 相机；
- STS3215 规格与 SDK；
- `genkiarm.urdf`；
- URDF 引用的 7 个 STL 网格或可测量的等效碰撞几何；
- 3D 模型或可测量连杆尺寸。

### 任务

1. 建立 J1–J5 与舵机 ID、URDF joint、控制命令通道及夹爪 ID6 的唯一映射；
2. 记录 6 个关节的真机零位、方向、软限位、最大安全速度和命令单位；
3. 修正 URDF 的 XML 兼容性，补齐 mesh 路径；禁止把缺少 inertial/collision/dynamics 的原始 URDF 直接当作动力学真值；
4. 测量连杆长度、末端偏置、夹爪 TCP；确认 J6 是夹爪整体姿态轴，并单独记录夹爪开合执行器；
5. 建立 FK/数值 IK，以至少 10 个非奇异姿态对照真机 TCP 测量；
6. 分别计算 intact、D2、D3、D4 的 position-only 共同可达域，以及含姿态约束的共同可达域；
7. 测量每个关节自由状态及 D2/D3/D4 锁定状态下的：
   - 稳态位置误差；
   - step response；
   - 回差；
   - 锁定角微动；
   - 命令—响应延迟；
   - 电流/温度可读性；
8. 确定 emergency stop；
9. 按底座、中间关节、腕部/夹爪姿态分别形成安全动作边界；
10. 决定 Reach/Push/Pick 的保留范围。

### 交付物

- `hardware/arm_spec.yaml`；
- `hardware/joint_map.yaml`（J1–J5、夹爪、URDF、舵机 ID、方向、零位、单位）；
- `hardware/calibration/` 原始数据；
- `hardware/safety_limits.yaml`；
- `sim/assets/genkiarm_calibrated.urdf` 与 `sim/assets/arm.xml` 初版；
- `reports/urdf-gap-report.md`（mesh、inertial、collision、dynamics 缺口）；
- `reports/g0-feasibility.md`；
- reachability 图与 target split。

### Pass

- 六关节命令映射无歧义，FK 端点误差满足任务容差需求；
- D2/D3 至少两个故障存在足够共同可达域；
- 锁定方式可重复且不导致危险电流/温升；
- 能在 10–20 分钟内重复安装和解除锁定；
- emergency stop 已测试。

### Block / Stop

- 锁定方式损坏舵机或不可重复；
- D2/D3 共同可达目标过少；
- 无法获得稳定关节状态；
- 无可靠急停。

## G1 — 最小机制验证

**建议日期**：2026-08-10 至 2026-08-23
**依赖**：G0 的尺寸、范围和最低物理参数  
**预算**：30–60 GPU-h；40–60 工时

### 固定范围

- Reach；
- D2、D3；
- 3 seeds；
- 4 方法：topology-only、residual-only/history、factorized、actor-head/parameter-matched；
- state observation；
- passive calibration；
- K = 0/1/2/5。

### V6 实际执行与裁决

- 原固定范围已完成用于路线裁决，但没有按原假设得到 factorized/few-shot Go；
- Reach 仅保留为环境与过拟合反例，不进入主结果；
- Push 成为机制任务，D2/D3、5 seeds、state observation、冻结部署已完成；
- K 曲线、residual latent 和 history/FiLM 分支均作为失败诊断归档，不补造正结论；
- 按预注册 Pivot 条款转为 robust zero-shot structured dynamics；
- parameter-matched 单模型公平对照、bootstrap 区间和训练耗时记录已完成。

### 交付物

以下为 V6 认可的实际 Pivot 交付包；原始路线失败项以审计报告交付，不要求
为了形式完整而重跑或制造正结果：

- 可运行的 MuJoCo Reach/Push 环境、修正后的夹爪接触模型与 100-step smoke test；
- corrected Push target split、dataset generator 和 D2/D3 接触/位移覆盖检查；
- conditional world model、topology encoder 和三成员 robust zero-shot ensemble；
- residual latent、history encoder、FiLM/residual correction 的实现与 No-Go 审计；
- 冻结 guarded MPC 及五目标控制审计；
- D2/D3、5 seeds、parameter-matched prediction table 与 seed bootstrap 区间；
- prediction error、ensemble disagreement、参数量和已测 wall-clock；
- checkpoint、run summary、日志、最终 JSON/CSV、复现指南和 Gate 报告；
- 自动测试通过记录及 data leakage/protocol correction 说明。

不再列为 G1 欠交付：原 K=0/1/2/5 的正向恢复曲线、原四方法在错误 Push
协议上的补跑、以及未通过统计门槛的 NR 控制主表。它们的科学结论均为
No-Go/不成立，后续只保留审计，不阻塞 G2。

### Go

以下是项目闸门，不是论文结论：

- D2、D3 中 factorized 相对 topology-only 的 NR 改善方向一致；
- 至少 2/3 seeds 改善；
- actor 冻结时收益仍存在；
- K 增加时 prediction error 与控制表现总体改善；
- 没有不可解释的 data leakage。

### Pivot

- factorized ≈ topology-only：转为 robust zero-shot / benchmark；
- actor-head update 才有效：转为 few-shot imitation；
- history encoder 明显更好：将 DFWM 降为 baseline，转向在线 diagnosis；
- WM 不稳定但 MPC dynamics 有效：改为 conditional dynamics + MPC，放弃 Dreamer 品牌。

### 实际 Gate（2026-08-19）

**Pivot-Go，可阶段交付。**判断依据：

- 修正协议后 D2/D3 均有真实接触和方块位移；
- 集成相对参数匹配单模型平均改善 30.7%，95% CI [15.1%, 42.6%]；
- D2、D3 和 5/5 seeds 改善方向一致；
- 模型与控制器在部署评估时冻结，无在线更新和不可解释的数据泄漏；
- 122 项自动测试通过，checkpoint、结果、日志和复现指南已归档。

限制：该 Gate 只批准进入 G2，不等同于论文创新成立。Guarded MPC 的扩展
审计区间跨零；结构化条件相对普通 deep ensemble 的独立贡献仍需在 G2 验证。

## G2 — 主会级仿真

**建议日期**：2026-08-24 至 2026-09-04
**依赖**：G1 Pivot-Go  
**预算**：60–100 GPU-h；60–90 工时

### V6 固定范围

- Push 为主任务；Reach 仅作边界和失败分析；
- D2/D3 为主故障，D4 仅在协议覆盖检查通过后加入；
- 5 training seeds，固定 corrected Push split；
- 核心方法：topology-conditioned ensemble；
- 强制基线：单 topology-only、参数匹配宽单模型、普通 deep ensemble、
  domain-randomized dynamics ensemble；
- 强制消融：去 topology condition、成员数 1/3/5、参数量与总训练 compute 对齐；
- held-out lock angle、摩擦、motor strength、backlash 及其组合；
- prediction 主指标使用多步 RMSE/NLL；不确定性只使用经校准验证的 ensemble
  disagreement，不使用当前反校准的 aleatoric log-std；
- 控制为次要验证：固定 guard，不允许按 test target 调阈值；
- 原 K=0/1/2/5 residual-calibration 曲线退出主线，仅作为失败结果保留。

### G2 Go

- 相对普通 deep ensemble 和 compute/parameter-matched 强基线仍有稳定收益；
- 五 seed 效应方向一致，seed-level 95% CI 不跨零；
- 在 held-out physics composition 上收益仍存在，不只来自平均多个随机初始化；
- 不确定性排序在 rollout-depth 分层后仍与误差正相关；
- 所有主表均由冻结配置自动生成，包含参数量、wall-clock、GPU 型号和失败 run；
- 若控制进入主张，其五 seed/多目标区间必须不跨零。

### G2 Pivot / Stop

- 与普通 deep ensemble 等价：降级为工程 benchmark，不主张结构化方法创新；
- 仅预测改善、控制无改善：论文定位为 dynamics prediction/uncertainty，删除恢复主张；
- held-out composition 优势消失：停止 ICRA 方法主线，转失败分析或更换问题设定；
- 结果依赖单一 seed、目标或协议调整：停止扩表并进行泄漏与选择偏差审计。

### G2 当前 Gate（2026-08-21，最终实验状态）

**结构化完整 world-model 主线已停止；保留 K1 约束关节模型与 ensemble uncertainty / selective prediction 主线。**

原始 DFWM 的 latent、encoder、FiLM、dropout 和 hypernetwork 均已五 seed 否定。CR-GWM
只确认 exact zero violation，公平性审计不能把预测收益归因于 reaction 结构。RC-GWM 在修复
学习率和数据多样性后仍有约 15% free-arm 退化，根因是错误移除锁定连杆的固定变换。

FT-GWM 用完整 SE(3) 链修复该物理错误。K0 运动学精确；K1 在 seed 7/17 均满足 violation=0
和 D3 free-arm 退化不超过 5% 的门槛，但仅有两 seed，且显式边计算未 compute-match。
K2 的 stop-gradient 成功保留 K1 joint fidelity，但 object RMSE 相对 matched graph 退化
`986.08%`，free-arm 相对 object-aware baseline 退化 `22.11%`，因此按预注册规则 **NO-GO**。
后续 FTC-WM Gate L 虽稳定收敛，但 object RMSE 平均约 `0.247`，相对 K2 v2 约恶化 `2.4x`，
同样 **NO-GO**。不再追加容量、接触特征、loss 权重或 epochs。五 seed ensemble/selective prediction 主表、
compute table 和最终 G2 synthesis 已生成；普通三成员 ensemble 相对 parameter-matched single
改善 `30.74%`，95% CI `[15.06%, 42.62%]`，但 structured vs ordinary 仅改善 `2.47%`，
CI `[-1.83%, 6.38%]`。50% coverage 的 RMSE 降幅为 `50.50%`，但存在 rollout-depth 混杂，
只主张 evaluated mixed-depth distribution 上的 selective rejection。权威汇总见
`reports/g2-final-synthesis-20260821.md`。下一决策是是否以这一收缩主张进入 G3。

### 交付物

- 主结果表；
- calibration curve；
- held-out composition 表；
- mechanism ablation；
- prediction/control correlation；
- robustness：backlash、delay、payload；
- failure taxonomy；
- compute table。

### Pass

- 主效应不是单一 seed 或单一 damage 驱动；
- held-out 组合上保持实质收益；
- 参数量与数据预算公平；
- 至少一个 negative/failure regime 被清晰识别；
- 完整运行成本不超过重新批准的预算。

### Stop

- factorization 在 G2 扩展后效应消失；
- damaged oracle 与 no-adapt 接近，说明任务本身不可恢复；
- baseline 无法公平实现；
- 仿真参数对结论极端敏感且不能由真机测量约束。

## G3 — 真机重复验证

**建议日期**：2026-08-24 至 2026-09-06，与 G2 并行
**依赖**：仅允许安全 adapter smoke 与 G2 并行；正式统计必须等待 G2 Go
**预算**：8–16 真机小时；8–16 GPU-h；30–45 工时

### 顺序

1. intact 与 D3 Push 安全/接口 smoke；
2. D3 单模型 topology-only；
3. D3 robust topology ensemble；
4. 第二 lock angle 或 D2；
5. 固定协议重复统计；
6. 只有前述稳定后才做扩展视频。

### 交付物

- 原始轨迹；
- calibration/evaluation 明确分离；
- 每次实验安全日志；
- 最低包每 condition ≥20 episodes；强证据包每 condition ≥30 episodes；
- 至少两个故障条件；
- 成功/失败视频；
- sim-to-real error report。

### Pass

- 至少两个 condition 中收益方向一致；
- 结果跨天或跨 target set 可重复；
- 未发生不可接受的安全事件；
- 真实 residual inference 确实降低 prediction error；
- 视频与数值结果一致。

## G4 — 论文与投稿

**建议日期**：2026-08-17 至 2026-09-15，与 G1–G3 并行
**依赖**：方法与实验设置可提前写；结果主张依赖冻结的 G2/G3 证据

### 交付物

- 重写后的 `paper/main.md` 与 `paper/main.tex`；
- 图表 source data；
- 补充视频；
- 代码 README；
- 环境与 checkpoint；
- integrity audit；
- submission check；
- 公开前隐私、安全和许可检查。
- 8 页完整稿页数预算、匿名检查、至少 3 个 ICRA 关键词与 PaperPlaza 元数据；
- 不超过 180 秒、20 MB 的匿名配套视频及上传回执。

### Pass

- 摘要、贡献和结论无预期结果；
- 所有数字可追溯到不可变结果文件；
- 表格与正文一致；
- closest work 更新到投稿前；
- 会议格式、匿名和页数符合官方规则；
- 代码/数据公开范围明确。

---

## 8. ICRA 2027 主目标倒排与硬闸门

ICRA 2027 是本轮唯一主目标。闸门用于控制主张和实验范围，而不是把会议重新降级为备选：

| 日期 | 必须完成 |
|---|---|
| 2026-08-08 | J1–J5/夹爪舵机映射冻结；原始 URDF 缺口清单完成；急停验证 |
| 2026-08-11 | G0 通过；校准后的 5-DoF MuJoCo arm 可加载；intact/D2/D3/D4 可达域完成 |
| 2026-08-16 | Reach intact/D3 baseline 稳定；数据管线可重跑；论文问题定义和方法初稿更新 |
| 2026-08-23 | G1 通过；factorized 在冻结 actor 时有方向一致收益；至少一个 D3 真机闭环 pilot |
| 2026-08-30 | D2/D3/D4 核心仿真表与公平基线完成；第二个真机 condition 开始统计；视频素材可用 |
| 2026-09-04 | 主文数值冻结；最低真机证据包完成；8 页匿名稿完整 |
| 2026-09-08 | 最终视频完成并在首个窗口关闭前上传；主图表与 failure cases 冻结 |
| 2026-09-10 | 内部审稿、引用/数字/匿名/格式核验完成 |
| 2026-09-12 | PaperPlaza 元数据、关键词、PDF compliance 预检查；预留三天缓冲 |
| 2026-09-15 | 11:59 PM PST 前正式提交 PDF |

若闸门未通过，首先降级非核心范围，而不是伪造完成状态：Push 可降为附加结果；5 seeds 可降至 3 seeds 并报告不确定性；K=2/10、latent-dimension 全扫描和 amortized encoder 可删除。以下行为始终禁止：

- 填预期数字；
- 把单次 demo 当统计；
- 删除失败结果；
- 弱化 baseline；
- 将 G1 小规模结果包装成完成的 G2；
- 用仿真结果替代真机主张，或把校准轨迹重复计入 evaluation。

### 8.1 Minimum Viable ICRA Submission

- 5-DoF+夹爪校准运动学与安全协议完整；
- Reach 主任务，D2/D3/D4 仿真，至少 3 seeds；
- topology-only、history encoder、matched descriptor、parameter-matched 四个公平基线；
- factorization 与 actor-freeze 两个关键消融；
- 至少两个真机故障条件，每条件不少于 20 个独立 evaluation episodes；
- 1–5 条校准轨迹与 evaluation targets 严格分离；
- 8 页匿名论文、可追溯表图和合规视频。

### 8.2 Strong ICRA Submission

在最低包之上增加：Push、5 seeds、每真机 condition ≥30 episodes、held-out composition 完整矩阵、prediction/control correlation、跨实验日重复和明确 failure regime。强包不得阻塞最低包按时冻结。

### 8.3 八页主文预算

| 内容 | 目标页数 |
|---|---:|
| 摘要 + 引言 + 贡献 | 1.0 |
| 相关工作 | 0.6 |
| 问题定义与 5-DoF 故障设置 | 0.7 |
| DFWM 方法 | 1.5 |
| 实验协议与真实平台 | 1.0 |
| 主结果、消融、真机结果 | 2.2 |
| 局限、结论、致谢 | 0.4 |
| 参考文献 | 0.6 |

总计目标 8.0 页；参考文献也计入上限。最终分页以官方模板编译结果为准。

---

## 9. 推荐代码结构

```text
robotarm/
├── pyproject.toml
├── README.md
├── PROJECT-PLAN-V4.md
├── config/
│   ├── base.yaml
│   ├── env/
│   ├── model/
│   ├── experiment/
│   └── splits/
├── src/robotarm/
│   ├── envs/
│   │   ├── protocol.py
│   │   ├── mujoco_env.py
│   │   ├── feetech_env.py
│   │   ├── tasks.py
│   │   ├── damage.py
│   │   └── safety.py
│   ├── models/
│   │   ├── topology_encoder.py
│   │   ├── residual_context.py
│   │   ├── world_model.py
│   │   ├── actor_critic.py
│   │   └── planner.py
│   ├── training/
│   │   ├── collect.py
│   │   ├── pretrain.py
│   │   ├── infer_context.py
│   │   └── evaluate.py
│   ├── baselines/
│   │   ├── topology_only.py
│   │   ├── history_encoder.py
│   │   ├── continuous_descriptor.py
│   │   └── parameter_matched.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── storage.py
│   │   └── validation.py
│   └── analysis/
│       ├── aggregate.py
│       ├── bootstrap.py
│       └── plots.py
├── sim/assets/
│   ├── genkiarm_source.urdf
│   ├── genkiarm_calibrated.urdf
│   ├── arm.xml
│   └── meshes/
├── hardware/
│   ├── arm_spec.yaml
│   ├── joint_map.yaml
│   ├── safety_limits.yaml
│   └── calibration/
├── scripts/
│   ├── smoke_test.py
│   ├── run_g1.py
│   ├── run_g2.py
│   ├── run_real.py
│   └── reproduce_paper.py
├── tests/
│   ├── test_kinematics.py
│   ├── test_env_contract.py
│   ├── test_damage_model.py
│   ├── test_context_shapes.py
│   ├── test_data_schema.py
│   └── test_determinism.py
├── runs/                 # gitignored
├── datasets/             # gitignored; manifest tracked
├── checkpoints/          # gitignored; manifest tracked
├── results/              # aggregate CSV/JSON tracked when final
├── reports/
├── experiments/
├── reviews/
└── paper/
```

### 9.1 工具选择

- Python 3.11 优先；若选用的 Dreamer 实现不兼容，则固定 3.10；
- 依赖统一写入 `pyproject.toml` 和 lock file；
- `pytest`；
- `ruff`；
- 类型检查至少覆盖数据 schema 与 env protocol；
- YAML + dataclass/Pydantic，避免早期引入复杂配置框架；
- TensorBoard + CSV/JSONL 为默认日志；W&B 可选，不作为复现依赖。

### 9.2 最低测试

- MuJoCo XML 编译；
- J1–J5、URDF joint、舵机 ID 与动作索引一一对应，ID6 单独作为夹爪；
- 六维 joint mask/action/state shape 固定，夹爪开合通道不得混入；
- reset/step 1000 步无 NaN；
- lock joint 的动作屏蔽与状态变化符合配置；
- FK 与 MuJoCo site position 一致；
- 同 seed 的短 rollout 可重复；
- 数据写入后可无损读回；
- context shape 与梯度范围正确；
- evaluation 不更新模型；
- calibration targets 不出现在 evaluation split。

---

## 10. 数据、日志与实验治理

### 10.1 轨迹 schema

每条 episode 至少包含：

```text
episode_id
timestamp_ns
platform: sim | real
task_id
target_id
split: calibration | validation | evaluation
damage_id
joint_mask                 # length 5: J1...J5
lock_angle
observation
action_commanded           # 5-DoF arm action; gripper_open separate
action_applied
gripper_open_command       # nullable independent actuator
next_observation
reward
success
done
safety_flags
hardware_state
camera_frame_ref
config_hash
git_commit
seed
```

### 10.2 不可变性

- 原始轨迹只追加，不原地编辑；
- 清洗生成新 dataset version；
- 每个 dataset 有 manifest、样本数、hash、来源和排除原因；
- 最终结果引用 dataset version 与 commit；
- calibration/evaluation split 创建后冻结。

### 10.3 Run 命名

```text
{stage}_{task}_{damage}_{method}_k{K}_seed{seed}_{yyyymmdd-hhmm}
```

每个 run 保存：

- resolved config；
- stdout/stderr；
- metrics JSONL；
- checkpoint；
- environment/system info；
- git commit；
- wall-clock、GPU 型号、peak memory；
- exit status。

### 10.4 结果发布规则

- 论文表格只能从 `results/final/*.csv` 自动生成；
- 手工复制数字必须二次核对；
- failed runs 不删除，写入 exclusion ledger；
- exclusion rule 在查看 test 结果前确定；
- 图表保留 source data。

---

## 11. 硬件与安全 SOP

### 11.1 开机前

1. 检查结构件、螺丝、线缆和电源；
2. 确认工作区无人和无易碎物；
3. 载入对应安全配置；
4. 验证急停；
5. 低速回零；
6. 读取温度、电压、电流和位置；
7. 相机与机械臂坐标标定检查。

### 11.2 锁定前

- 断电或进入安全模式；
- 记录锁定关节与角度；
- 按固定机械步骤安装；
- 拍照记录；
- 手动小幅加载，确认微动范围；
- 不使用“继续加扭矩直到不动”的不可控方式。
- D2/D3/D4 分别使用经过验证的专用固定位置与夹具；不得把适用于某一中间关节的锁定力矩直接复制到其他关节；
- J1 底座、J5 腕部和夹爪故障不进入主实验，除非重新完成工作区、碰撞和末端姿态安全评审。

### 11.3 运行中

- 操作者始终在场；
- 先执行低幅安全动作；
- 任何软限位、通信超时、异常电流、异常温升触发 stop；
- 阈值必须来自数据手册或 G0 实测，禁止在计划中虚构固定数字；
- 每个 condition 之间进行冷却与外观检查。
- 对六个关节分别记录 commanded/applied position；锁定关节出现超出 G0 微动范围的位移时立即停止；
- 夹爪开合执行器与 J6 姿态命令独立限幅，避免通道映射错误造成夹持或碰撞风险。

### 11.4 结束后

- 保存完整日志；
- 记录急停、超限和人工干预；
- 检查舵机温度与松动；
- 生成 session summary；
- 不把发生人工干预的 episode 混入普通成功率。

---

## 12. 资源分配

| 资源 | 角色 | 不承担 |
|---|---|---|
| MacBook | 代码、测试、MuJoCo smoke、真机控制、分析、论文 | 长时全量预训练 |
| RTX 3070 | 小模型、单 cell、消融 smoke、回归测试 | 大 batch 全主表 |
| 云 RTX 4090 | WM pretraining、G1/G2 主实验 | 未通过 smoke 的调试任务 |
| 实体臂 | G0 测量、集成、G3 正式评估 | 没有安全限制的探索 |

### 12.1 预算明细

| 类别 | G0 | G1 | G2 | G3 | G4 | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| 工程/研究工时 | 16–24 | 40–60 | 60–90 | 30–45 | 30–45 | 176–264 |
| GPU-h | 0–5 | 30–60 | 60–100 | 8–16 | 0–5 | 98–186 |
| 真机小时 | 4–8 | 0–2 | 0 | 8–16 | 0–2 | 12–28 |
| 存储增量 | <5 GB | 20–40 GB | 40–150 GB | 15–35 GB | <10 GB | 75–240 GB |

### 12.2 预算审批点

- 单 run 超过预计时间 2 倍：暂停批量；
- G1 超过 60 GPU-h 仍无机制信号：强制评审；
- G2 预计超过 100 GPU-h：必须删除次要消融或重新批准；
- 真机发生一次严重安全事件：暂停，完成 root-cause report 后再恢复。

---

## 13. 时间表与关键路径

### 13.1 ICRA 六周倒排路线

| 周 | 日期 | 主任务 | 硬输出/降级规则 |
|---|---|---|---|
| W0 | 08-06–08-09 | J1–J5+夹爪映射、URDF 修复、FK、急停与测量模板 | joint map、URDF gap report、安全边界 |
| W1 | 08-10–08-16 | 5-DoF MuJoCo、可达域、Reach env、数据管线、intact/D3 baseline | G0 通过；健康 baseline 不稳则暂停方法扩展 |
| W2 | 08-17–08-23 | residual latent、factorized G1、3 seeds、D3 真机 pilot、方法稿 | G1 无信号则缩主张或 Pivot，不伪装结果 |
| W3 | 08-24–08-30 | D2/D3/D4、四基线、held-out 核心表；第二真机 condition；视频采集 | Push、K=2/10 和非核心消融可降级 |
| W4 | 08-31–09-06 | 核心消融、真机正式统计、主结果冻结、完整 8 页稿 | 09-04 冻结数字；未完成的强包项目删除 |
| W5 | 09-07–09-13 | 视频上传、failure analysis、内部审稿、引用/数字/匿名/PDF QA | 09-10 完成审计，09-12 完成预提交 |
| Submit | 09-14–09-15 | 最终检查、PaperPlaza 上传与回读 | 不在最后一小时首次上传 |

### 13.2 关键路径

```text
硬件测量
  -> 可达域与任务冻结
  -> MuJoCo 环境
  -> 健康/损坏 baseline
  -> G1 原始假设 No-Go
  -> robust zero-shot Pivot-Go
  -> G2 强基线与 held-out composition
  -> G2 Go
  -> 真机正式统计
  -> 论文结果冻结
  -> integrity/submission check
```

### 13.3 可并行任务

- 环境实现与文献监控；
- 仿真批量与真机 adapter 开发；
- G2 运行与图表脚本；
- 真机采样与论文方法重写；
- artifact QA 与投稿格式检查。

不能并行绕过的依赖：

- 没有 G0 不冻结 task；
- 没有 G1 不跑 G2；
- 没有稳定 G2 不做正式真机大样本；
- 没有冻结结果不写结论。

---

## 14. 第一阶段逐日任务

### Day 1

- 创建 `pyproject.toml`、`src/`、`tests/`；
- 固定 Python 版本选择流程；
- 写 `RobotEnv` protocol；
- 建立 `hardware/arm_spec.yaml` 与 `hardware/joint_map.yaml` 模板；
- 冻结 J1–J5、夹爪、URDF joint、舵机 ID、动作索引和功能角色；
- 保存当前 git 状态，不触碰现有未跟踪文件。

### Day 2

- 补齐 URDF mesh 路径，记录 inertial/collision/dynamics 缺口；
- 测量六关节连杆、TCP 与零位；
- 实现 FK；
- 写 FK 单元测试；
- 定义 D2/D3/D4；
- 草拟 safety limits。

### Day 3

- 构建最小 MJCF；
- 加载、reset、step；
- 校验 MuJoCo site 与 FK；
- 输出 intact reachability。

### Day 4

- 输出 D2/D3/D4 reachability；
- 生成共同目标集合；
- 决定 Pick 是否删除；
- 写 G0 中期报告。

### Day 5

- 真机位置响应与延迟测量；
- 自由状态回差测量；
- 急停测试；
- 修订安全配置。

### Day 6

- 安装锁定机构；
- 测量锁定角微动、摩擦代理量和重复性；
- 拍照与保存原始数据；
- 如出现风险立即停止。

### Day 7

- 完成 G0 gate review；
- 冻结 Reach target split；
- 更新 MuJoCo parameter ranges；
- 决定是否进入 G1。

### Day 8–10

- 实现 Reach reward/success；
- 轨迹 schema 与 validation；
- topology-only policy/WM baseline；
- 1000-step 与短训练 smoke。

### Day 11–14

- residual latent optimization；
- factorized conditioning；
- frozen actor/MPC；
- 第一个 D3、K=0/1/5、seed 0 对照；
- 根据真实 wall-clock 更新 G1 预算。

---

## 15. 风险登记表

| ID | 风险 | 概率 | 影响 | 早期信号 | 缓解 | 触发决策 |
|---|---|---|---|---|---|---|
| R0 | URDF 与真机不一致 | 高 | 致命 | FK/TCP、轴向或零位明显偏差 | joint map；10+ 姿态校验；校准后再转 MJCF | 阻塞 G0 |
| R1 | 锁定后任务不可达 | 中 | 致命 | IK/采样可达率低 | 使用共同可达域；删 Pick | Stop/缩范围 |
| R2 | factorization 无收益 | 中高 | 致命 | G1≈topology-only | 转 benchmark/zero-shot | Pivot A |
| R3 | actor BC 才有效 | 中 | 高 | actor frozen 无提升 | 转 few-shot imitation | Pivot B |
| R4 | history encoder 更强 | 中 | 高 | 短历史已识别全部变化 | 转未知损坏 diagnosis | Pivot |
| R5 | WM 训练不稳定 | 中高 | 高 | loss/return 高方差 | 小 RSSM；MPC；复用可靠实现 | 架构降级 |
| R6 | sim-to-real gap 过大 | 高 | 高 | 真机 prediction error 不降 | 扩 residual ranges；实测 actuator model | 降低主张 |
| R7 | 硬件损坏 | 低中 | 高 | 温升、电流、松动 | 安全边界、备件、监督运行 | 暂停 |
| R8 | 计算超预算 | 中 | 中 | run >2×预计 | early stopping、缩 seed/cell | 预算复审 |
| R9 | competitor 抢先 | 中 | 高 | 2026 新预印本重合 | 月度监控；调整 novelty delta | idea 复评 |
| R10 | ICRA 截止诱发低质量提交 | 高 | 高 | 9/5 仍无冻结结果 | 硬退出门 | 转 RSS/CoRL |
| R11 | 单平台证据不足 | 高 | 中高 | reviewer 认为 demo-only | 多 condition、多日、开源 protocol | 系统贡献增强 |
| R12 | 统计功效不足 | 中 | 高 | CI 极宽 | 增 evaluation episodes；报告效应量 | 克制结论 |
| R13 | 数据泄漏 | 中 | 致命 | calibration/eval targets 重合 | split hash、自动检查 | 重跑 |
| R14 | 论文旧叙事残留 | 高 | 高 | binary>continuous 仍出现 | G2 后全稿重写而非局部替换 | integrity audit |

---

## 16. 论文与证据同步计划

### G0 后可写

- 平台与故障定义；
- 可达域构造；
- 安全与测量协议；
- 任务范围。

### G1 后可写

- DFWM 方法；
- context factorization；
- deployment inference；
- G1 作为内部证据，不一定进入最终主表。

### G2 后可写

- 实验设置；
- baseline 公平性；
- 主结果、消融、held-out composition；
- failure cases；
- compute。

### G3 后可写

- 真实平台结果；
- sim-to-real 分析；
- 安全与 wall-clock；
- 局限。

### 必须整体删除或重写的旧内容

- binary vs continuous 的表达能力论证；
- random mask “保证任意损坏 in-distribution”的泛化表述；
- token + actor-head 联合微调作为核心机制；
- 预期 60%/80%、15 points 胜幅；
- 500 GPU-h 与碳排的未实测数字；
- RSS 2026 仍可投稿的时间线。

### 论文 owner 顺序

1. `ccf-experiment-designer`：G0 结果出来后冻结完整实验协议；
2. `ccf-paper-writer`：G1 通过后重写问题和方法；
3. `ccf-visual-composer`：G2/G3 真实数字冻结后生成图表；
4. `ccf-integrity-auditor`：核查 claim、数字、引用和表图；
5. `ccf-paper-reviewer`：模拟主会评审；
6. `ccf-submission-checker`：官方 CFP 发布或投稿前核验。

---

## 17. 项目管理节奏

### 每日

- 今天完成了什么；
- 新生成哪些 artifact；
- 哪个假设被支持/削弱；
- GPU/真机消耗；
- blocker；
- 明天唯一最重要任务。

### 每周

```text
Week:
Gate:
Completed:
Evidence:
Failed/Excluded runs:
Budget used:
Risks changed:
Decision:
Next week:
```

### 每个阶段门

必须留下：

- 输入版本；
- 运行列表；
- pass/fail 证据；
- 预算实际值；
- Go/Pivot/Stop 决策；
- 决策人和日期；
- 下一阶段范围。

---

## 18. Artifact 清单

### G0

- [x] arm spec
- [x] safety limits
- [x] calibration raw data
- [x] reachability split
- [x] MuJoCo XML
- [x] feasibility report

### G1

- [x] reproducible environment
- [x] corrected Push protocol and data generator
- [x] topology encoder and conditional world model
- [x] residual context/history/FiLM diagnostics（No-Go，已归档）
- [x] robust zero-shot topology ensemble
- [x] parameter-matched five-seed prediction audit
- [x] frozen guarded MPC audit（次要结果，区间跨零）
- [x] checkpoint、manifest、日志、结果表和复现指南
- [x] gate report

### G2

- [x] immutable G2 configs and preregistered exclusions
- [x] ordinary deep-ensemble baseline（g2_push_ensemble_v1，5 seeds）
- [x] domain-randomized ensemble baseline（g2_domain_randomized，5 seeds）
- [x] held-out topology experiment（g2_push_heldout_topology_v1，5 seeds）
- [x] GRU hidden-state conditioning probe（probe_conditioning_collapse.py）
- [x] gate reports（ordinary-ensemble-gate, heldout-topology-gate）
- [x] bootstrap 95% CI 两轮实验
- [x] failure analysis（conditioning redundancy + weak zero-shot generalization）
- [x] 分歧指纹拓扑识别（route2_topo_id，5 seeds，100% K=1 准确率）
- [x] 选择性预测（selective prediction -51% RMSE @ 50% coverage，5 seeds）
- [x] DFWM-Hypernetwork OOD 审计（5 seeds No-Go；K-shot 独立贡献近零，作为失败路线归档）
- [x] MuJoCo Warp GPU 批量采集（collect_warp.py，63x 加速）
- [x] shared chain graph dynamics 与 topology-surgery 消融（graph 架构有效；topology surgery 单独无稳定预测收益）
- [x] CR-GWM Gate E（D3 held-out，5 seeds，provisional pass；zero violation）
- [x] Gate F fairness audit（seed 7；未通过 matched-capacity 预测归因）
- [x] Gate G direct-projection audit（原 hidden 容量错误已更正；matched projection 零 violation，free-arm 约退化 5.29%）
- [x] Gate H：matched graph + gated reaction head（统一指标后 provisional pass；不再扩展）
- [x] Gate I 五 seed RC-GWM（仅 2/5 通过；稳定模型 No-Go）
- [x] J1--J6 RC-GWM 逐原因诊断（优化/数据修复后仍有 reduced-coordinate 归纳偏置）
- [x] Gate K0 固定 SE(3) 运动学（PASS）
- [x] Gate K1 FT-GWM 自由关节动力学（seed 7/17 provisional pass；zero violation）
- [x] Gate K2 隔离 object/contact head（有效 v2 NO-GO；停止完整 world-model 分支）
- [x] FTC-WM Gate L contact/free-object 分支（NO-GO；object rollout 较 K2 v2 约恶化 2.4x）
- [x] K0--K2 审计报告（g2-fixed-transform-graph-gate-k1-20260820.md）
- [x] 最终 G2 synthesis：ensemble/selective prediction 主表、compute table 与论文主张冻结（`reports/g2-final-synthesis-20260821.md`）

### G3

- [ ] real raw trajectories
- [ ] safety ledger
- [ ] two fault conditions
- [ ] 最低包 ≥20 episodes/condition；强包 ≥30 episodes/condition
- [ ] videos
- [ ] sim-to-real analysis

### G4

- [ ] revised manuscript
- [ ] source-data figures
- [ ] references verified
- [ ] artifact README
- [ ] checkpoint manifests
- [ ] integrity audit
- [ ] simulated review
- [ ] official submission check

---

## 19. `ccfa.yaml` 建议更新

按 orchestrator 规范，本轮不自动改写 `ccfa.yaml`。建议在用户明确批准项目状态迁移后：

```yaml
project:
  title: 六自由度低成本机械臂关节锁定后的损坏因子化世界模型与少样本安全恢复
  revised: 2026-08-06

target_venue:
  primary: ICRA-2027
  fallback:
    - RSS-2027
    - CoRL-2027
  submission_deadline: 2026-09-15T23:59:00-08:00
  rule_verified: 2026-08-06
  deadline_policy: verify-official-again-before-submission

stage: v4-6dof-plan-ready-g0-started

claims:
  central_claim:
    statement: 已知离散损坏拓扑与未知连续残余动力学的因子化，可降低低成本机械臂关节锁定后的真实校准数据需求
    status: needs-g1-mechanism-evidence
  sub_claims:
    - statement: actor 与 world model 冻结时，residual context inference 仍带来恢复
      status: needs-g1
    - statement: factorization 可泛化到 held-out topology-physics 组合
      status: needs-g2
    - statement: 方法在至少两个真实锁定条件下可重复
      status: needs-g3

artifacts:
  - reviews/idea-review-robotics-topvenue-20260730.md
  - PROJECT-PLAN-V4.md
```

旧实验 E1/E2/ABL/ROB 不应直接删除；建议标记为 `superseded-by-v4-gates`，保留历史审计。

---

## 19.1 BT-DPWM 固定路线更新（2026-08-21）

核心方法已固定为 **Block-Triangular Damage-Projected World Model（BT-DPWM）**，
不再切换到新的方法名称。双专家不是废弃路线，而是 BT-DPWM 中 robot/object
独立表示的来源；BT-DPWM 进一步加入有向耦合、解析损伤投影和可执行梯度边界。

X0--Y6 逐步门控已经完成。严格 object-agnostic robot、全程梯度手术、末段
joint-only refinement 和同时双 horizon 更新均被实验否决。当前唯一保留版本 Y6：

1. contact-conditioned robot graph 以 rollout horizon 10 训练；
2. 解析投影在每一步强制锁定关节位置/速度约束；
3. 冻结 robot block；
4. independent recurrent object graph 在固定 robot rollout 上以 horizon 5 训练；
5. object loss 不能反向更新 robot block。

Seed 7 对 frozen compute-matched shared graph 的 depth-10 结果：free-arm +4.16%，
object +3.69%，overall +4.16%，constraint violation RMS=0，达到预注册门槛。
冻结 seed 17/27 复现也通过；三 seed 均值为 free-arm +7.90%、object +5.30%、
overall +7.88%，3/3 per-seed PASS。该结果升级为 primary-domain 三 seed mechanism
PASS；在 compute/parameter audit、跨损伤域和公开强基线比较完成前，不升级为
完整论文主张。完整审计见
`reports/g2-bt-dpwm-gates-x0-y6-20260821.md`。

下一步严格限制为：多 seed 复现 -> compute/parameter audit -> DFWM 同协议比较 ->
通过后才进入真机 smoke/G3。任何新改动必须针对复现中观察到的具体失败，不再新建核心框架。

**后续公平性审计更新：** Z0 对 h96/120 shared 的四测试域三 seed 均值仍为正
（free +6.20%、object +2.21%、overall +6.15%），但 12 个 cell 中有两个轻微
overall 退化，严格门槛 NO-GO。Z1 进一步训练参数匹配 h136 shared 240 epochs；
BT-DPWM 相对该强基线三 seed 平均为 free -39.02%、object +36.40%、overall
-36.19%，0/3 PASS。故不得再宣称当前 Y6 overall 超过公平强基线。保留的机制
证据是 object block 的稳定优势；下一步只允许在固定约 338k 总参数预算内把容量
从 object block 重分配给 robot block，再重复相同 Z1，禁止切换核心框架。

在容量改动前完成 Y7 robot-budget 归因：保持结构不变，将 robot h10 训练从
120 增至 240 epochs，object 仍为 120/h5。robot train loss 从 0.02291 降至
0.01108，但 seed 7 主域相对 h136/240 为 free -72.94%、object +46.57%、overall
-64.09%，NO-GO；因此停止 seed 17/27，排除“仅更新次数不足”。容量分配仍只是
待验证假设。下一步须先区分 robot 表示/conditioning 归纳偏置与缺少 contact
辅助监督，不得继续追加 epoch。

Z2 冻结诊断进一步排除 contact representation 与 object-feedback drift：hidden
contact AUC 为 shared 0.996、Y6 0.990、Y7 0.995；true-object oracle 对 Y6 depth10
free RMSE 仅从 0.3272 改到 0.3220。Z3 的同宽 h96/240 shared 对照中，Y6 四域
均值为 free +0.14%、object +7.08%、overall +0.25%。因此 h96 robot 本身并未因
梯度边界或 joint-only 监督显著落后；Z1 的巨大差距主要在 shared 扩宽到 h136
后出现。容量分配假设获得直接支持但尚未最终证明；下一门必须固定总参数约
338k，增大 robot width、缩小 object width，不得增加总容量或 epochs。

固定预算 Z4--Z5 更新（2026-08-22）：随机 robot128/object56（337,518 params）
在 seed7 主域 free -90.13%，NO-GO。改为从 h136/240 shared 复制并冻结稳定
robot scaffold、丢弃其 object head、训练独立 object32 后，Z4b（336,910 params）
seed7 主域 overall +6.35%，四域均值 free +0.65%、object +25.57%、overall
+2.12%。加入零初始化 rank-8 reaction adapter 的 Z5 总参数338,056（比baseline少46），
seed7 四域 overall +5.03%，但三seed×四域最终为 free -0.36%、object +22.00%、
overall +0.59%，4/12退化，NO-GO。最终goal（overall均值至少+5%、最多1个退化）
尚未完成。下一步仅允许同架构、同参数的 validation-selected reaction，并必须把
zero-reaction checkpoint 纳入选择，禁止增参或换核心框架。

Z6--Z8 稳定化更新（2026-08-22）：Z6 以三 validation domain 每两轮选择
reaction checkpoint，并显式包含 epoch0。seed7/17/27 分别选择 40/0/40；冻结后
三 seed×四域为 free -0.71%、object +22.00%、overall +0.24%、6/12 退化，
NO-GO。seed27 validation 改善但 test 退化，证明当前 validation-free 指标不能可靠
选择 reaction。

reaction 输出诊断显示 contact/non-contact 范数比仅为 1.18/0.76/0.99；adapter
没有事件选择性。解析 pusher-box gap 则跨种子稳定分离：contact gap 中位数约
-11.5 mm、90% 分位约 -8.0 mm，non-contact 10% 分位约 -1.5 mm。Z7 因而加入
零参数解析 soft gate（总参数仍为 338,056）。带门重训的 seed7 四域为 free
+2.34%、object +25.57%、overall +3.73%、0/4 退化；方向正确但未达 +5%。对
冻结 Z5 adapter 扫描后锁定 threshold=+5 mm、temperature=2 mm，三 seed×四域
为 free +0.77%、object +21.99%、overall +1.68%、5/12 退化，仍 NO-GO。

Z8 将同一 gated adapter 改成 teacher-forced one-step residual training，seed7 四域
仅 free +0.80%、object +25.54%、overall +2.27%、1/4 退化，亦 NO-GO。因此当前
问题已经收窄为：reaction 需要接触事件触发后的有限衰减记忆与跨 seed 尺度约束；
纯当前几何门、纯 validation checkpoint 或纯一步残差均不足。下一实验只能在同一
rank-8 reaction 模块内实现 bounded event-memory，并以 Z4b zero-reaction、Z5
ungated、Z7 current-gate 为严格消融；不得扩充参数预算或更换 BT-DPWM。

Z9--Z12 reaction 归因更新（2026-08-22）：Z9 对冻结 Z5 做统一幅度扫描，seed7
在 scale=0.75 时 overall +5.12%，但锁定到三 seed 后仅 free +0.06%、object
+21.99%、overall +0.99%、4/12 退化；统一幅度不是跨 seed 解。scale=0 的三 seed
overall 仅 +0.75%，说明独立 object 的稳定收益本身不足以达到 overall +5%，必须
同时获得可重复的 robot 改善。

Z10 将 reaction 输入从 seed-specific 136-D robot hidden 改为 10-D 每关节物理特征，
用 hidden80 后 adapter 约 1,042 参数，仍小于 Z5 的 1,146 参数。seed7 冻结部署
scale=0.30 时 free +3.94%、object +25.60%、overall +5.25%；但三 seed 固定规则
仅 free +0.86%、object +21.97%、overall +1.76%、4/12 退化。分项显示 seed7
mixed-unseen 仍为 overall -6.79%，故随机 latent basis 只是部分原因，主要失败转为
held-out physics 下的 correction 过拟合。

Z11 对 12 个训练域使用平滑 worst-domain reaction objective，seed7 scale=0.30 为
overall +5.31%，但 mixed-unseen 仅从 -6.79% 改至 -6.35%，改善过小且逐域小 batch
带来约 12 倍吞吐损失，停止跨 seed 扩展。Z12 加入零参数解析 event trace，连接
Z7 current-gate 与 Z5 ungated；threshold=+5 mm、decay=0.95 的 seed7 四域为 free
+3.41%、object +25.65%、overall +4.72%、0/4 退化，未达 +5%。因此 event-memory
能降低退化但不能单独解决跨 seed/held-out physics。下一门必须提供可部署的
reaction confidence/stability constraint，并明确证明相对 zero-reaction 不劣；禁止
用 test seed 分别选择 scale、epoch 或 gate。

Z13--Z14 监督与随机性归因（2026-08-22）：先计算 object-only 理论上界。若 robot
完全等于强 shared baseline 且 object RMSE 降为 0，12 cell 的 overall 改善均值也
只有 +2.26%；因此 goal 的 +5% 必须包含约 3 个百分点的稳定 robot 改善，不能再
把 object 优势当作充分条件。

Z13 用训练轨迹 MuJoCo contact mask 加权 physical reaction 的一步残差监督，部署仍
只用解析 geometry gate。seed7 在 scale=0.25 已为 free -0.73%、overall +0.83%，
更大 scale 单调恶化，NO-GO。结论是 object contact 事件不等于可泛化 joint reaction
监督；Z5 的正收益来自更广义的 rollout correction，不能收缩成纯碰撞冲量。

Z14 固定 physical adapter 第一层随机 feature basis（所有 seed 同一确定初始化），
排除随机参数坐标。seed7 在 scale 0.30/0.40/0.50 形成 +5.14/+5.29/+5.00% 的宽
有效区间；锁定 scale=0.40 后三 seed×四域仍仅 free +0.84%、object +21.98%、
overall +1.74%、4/12 退化，NO-GO，几乎等同 Z10。故跨 seed 方差来自各训练集与
scaffold residual 的不一致，而不是 adapter 随机初始化。下一步禁止继续调
scale/gate/seed；必须引入不依赖 test selection、能约束多步 robot residual 方向的
训练信号，同时保持 BT-DPWM、参数上限和三 seed 独立性。

Z15--Z17 多步安全与拓扑机制审计（2026-08-22）：Z15 引入 paired dominance
reaction loss，在每条训练轨迹/每个 rollout depth 与同模型 zero-reaction 对照，对
任一误差上升施罚。weight=10 的 seed7 overall +2.62%；weight=2 的 seed7 free
+3.53%、object +25.58%、overall +4.88%、1/4 退化。锁定 weight=2 后三 seed×
四域仅 free +0.31%、object +21.96%、overall +1.24%、7/12 退化，NO-GO。
seed17/27 validation 均改善但 test 仍退化，证明训练轨迹 dominance 不构成分布外
安全保证。

随后发现 V0--Z15 的统一训练/evaluate 路径虽然生成真实 damage mask，却向模型内部
传入全零 mask/angle，只在模型外做最终 projection；因此 BT-DPWM 的内部 action
屏蔽、逐步 projection 和 topology message 从未被激活。Z16 从 h136/240 scaffold
初始化后全量 topology-aware fine-tune 40轮，robot train loss下降约42%，但 seed7
primary free -101.47%，显示 D2/D4 全量适配破坏 held-out D3 scaffold，NO-GO。

Z17 仅训练 robot encoder 中此前无梯度的 mask/lock-angle 两个输入列，共272个现有
权重且从零初始化，其余318k robot权重冻结；zero-mask 前向保持 scaffold。seed7
primary overall +6.50%，四域 free +2.17%、object +25.81%、overall +3.58%、1/4
退化；但三 seed最终为 free -2.40%、object +22.01%、overall -1.36%、7/12退化，
NO-GO。内部 topology 是此前遗漏的核心机制，但在 leave-D3-out 数据下仍不能提供
可重复 robot 改善。下一步不得把任何单 seed topology 结果升级为主张。

Z18 解析关节积分审计（2026-08-22）：缓存轨迹中 `q[t+1]-q[t]` 对 current/next/
average qvel 的最优标量步长均约0.005 s；半隐式积分的位置增量 residual RMSE约为
原增量RMSE的29.4%，支持加入无参数运动学约束。在冻结 Z5 zero-reaction scaffold
上，将 learned q 与 `q+0.005*qvel_next` 融合，seed7 四域 overall 从 blend0 的
+2.12% 平滑升到 blend0.75 的 +2.51%，1/4退化；纯积分为+2.48%。锁定blend0.75
后三seed×四域为 free -0.37%、object +21.96%、overall +0.58%、6/12退化，
NO-GO。解析积分关系真实但不足以跨seed改善强baseline，不进入最终组件。

Z19 双对象路径审计（2026-08-22）：过滤掉弱 h96 baseline 后，Z4--Z18 已有候选的
12-cell oracle 组合上界为 free +4.45%、object +21.99%、overall +5.21%，但仍有
2个退化 cell；目标在数值上接近可达，但现有机制间没有可部署统一选择规则。

为隔离 independent object rollout 对 frozen robot scaffold 的输入分布漂移，Z19 在
同一 BT-DPWM 内加入 rank-8 shadow object context：shadow 只供 robot 条件化，独立
object expert 仍输出最终 object。shadow head共1,164参数，总模型338,074，比
h136/240 baseline少28；robot完全冻结，shadow joint-rollout训练40轮后再训练
object120轮。seed7 primary为free +4.71%、object +42.14%、overall +8.09%，四域
为free +2.35%、object +25.55%、overall +3.71%、1/4退化。锁定后三seed×四域为
free +0.17%、object +21.96%、overall +1.10%、4/12退化，NO-GO。双对象路径能
减轻object-feedback drift但不能制造跨seed稳定robot优势；不得以seed7 primary
升级方法主张。

Z20 固定预算 robot ensemble（2026-08-22）：两个独立 h96 robot graph experts 加
单一 object32 共337,448参数，比 h136/240 baseline少654；每步平均 robot transition
后解析投影。robot按公平240 epochs/h10训练，loss从0.1790降至0.01070，低于强
baseline训练loss约0.0176；随后object120 epochs/h5降至0.000726。然而seed7
primary相对强baseline为free -30.20%、object +42.01%、overall -24.31%，NO-GO，
停止seed17/27。总参数/训练loss匹配不等于分布外泛化；从头训练的双窄专家仍显著
不如预训练宽h136 scaffold。下一步必须保留强scaffold，不再用窄ensemble替代。

Z21 h136 robot-head-only适配（2026-08-22）：保留强baseline的encoder/message/
temporal，仅用h10 joint rollout和lr=1e-4微调18,906个robot-head现有权重40轮，
再冻结并训练object120轮。robot train loss从0.02162降至0.01005，object loss降至
0.000872；但seed7 primary为free -60.00%、object +42.08%、overall -52.14%，
NO-GO，停止seed17/27。即使低维、低学习率的监督式head重拟合也会破坏held-out
泛化；后续必须完整保留h136 robot scaffold，不再用当前D2/D4训练集更新其动力学
权重。

Z22--Z26 scaffold 选择与结构约束审计（2026-08-22）：Z22 对同一 h136/240
轨迹最后80个 checkpoint 做 SWA，seed7 primary object +42.07%，但 free -84.11%、
overall -74.72%；固定学习率下的参数平均离开了可部署 robot basin。Z23 在冻结
validation split 比较 final/EMA/逐轮最优，选择 final epoch240，仍为 free -93.50%、
overall -83.53%。这证明 shared validation loss 不能评价复制到 BT-DPWM 后的
leave-D3 robot rollout。

Z24 引入零参数 analytic contact-gated object→robot context。审计发现原解析 gate
只看 XY，D3 中将机械臂垂直越过方块误判为接触；修正为完整3D capsule-box距离后
三 split precision 达100%，但旧 -5 mm 阈值 recall过低。仅由train/validation
选择0 mm后，D3 contact F1为0.62；旧权重阈值探针仍为四域overall -24.50%，不
重训扩展。Z25 从第一轮起训练显式 topology-conditioned h136/240 scaffold，最终
validation 0.00724，但primary free -128.50%、overall -116.37%。Z26 从训练起执行
半隐式 `q_next=q+0.005*v_next`，validation进一步降至0.00680，primary仍free
-92.24%、overall -82.35%。Z23--Z26 一致说明：当前D2/D4数据上任何 robot
scaffold 重识别，即使训练/validation更优和结构关系正确，都会破坏held-out D3；
后续不得再以降低训练loss为理由重训h136主体。

Z27--Z30 冻结 scaffold 小修正审计（2026-08-22）：Z27 在 Z4 上增加22参数的
关节共享线性物理残差，用闭式ridge拟合并由普通validation选λ=1，primary free
-57.29%、overall -49.60%。Z28 改为训练集内部 leave-one-topology-out 选择，λ
提高到1000，仍为free -52.64%、overall -45.26%；已见拓扑残差不能外推D3。
Z29 将development seed7的同一physical adapter权重部署到三seed各自scaffold，
三seed×四域为free -4.69%、object +21.96%、overall -3.63%、8/12退化，排除
adapter随机训练坐标是主因。Z30 对frozen scaffold做零参数q/v delta contraction，
seed7最佳position scale=0.90也仅四域overall +2.52%，未达到+5%。因此additive
correction、跨seed共享correction与简单contractive scaling均冻结为NO-GO。

当前最强诚实结论仍是Z4 frozen-scaffold/object-specialization：跨seed object收益
稳定，但overall不足。达到+5%必须产生真正可迁移的robot改善；现有12-cell候选
oracle仅+5.21%且仍有2个退化cell。下一实验若继续，只允许预先定义、部署可用的
状态级confidence/selection机制，并必须包含zero-correction安全路径；禁止继续
重训scaffold、调test阈值或添加未经归因的新专家。

Z31 状态级 trust-region 审计（2026-08-22）：对 Z5 reaction 增加零参数逐关节
相对上界，修正范数不得超过 frozen scaffold 自身 `(Δq,Δv)` 范数的固定比例；
scaffold增量为零时严格回到zero-correction。seed7预声明扫描从clip0的overall
+2.12%单调上升，在clip0.8达到free +3.76%、object +25.66%、overall +5.07%、
1/4退化，随后锁定0.8。三seed×四域复现仅free -0.34%、object +21.99%、
overall +0.60%、4/12退化，NO-GO。相对trust region能限制幅度，不能修正seed27
adapter的错误方向。

**BT-DPWM 固定预算目标当前阻塞**：Z4--Z31 已覆盖 object specialization、低秩/
物理/确定性/共享/线性 reaction、geometry/event gate、paired safety、内部topology、
解析积分、shadow context、ensemble、scaffold选择及state-level trust region。共同
证据是object稳定改善约22%，但robot correction在未见D3与seed27上无可部署选择
信号；object-only理论overall上限仅+2.26%。若不改变核心BT-DPWM，解除阻塞至少
需要一种新的信息来源：训练split加入不与test重合的第三关节锁定元训练域、真机/
仿真可测的在线接触或短时校准，或预注册更多训练seed用于group-robust识别。任何
一项都会扩展当前固定协议，须由项目负责人明确授权后另立gate；不得把现有结果
表述为已达到+5%。

Z32 协议扩展与公平重基线（2026-08-22）：经负责人授权，在不改变四个冻结测试域的
前提下，训练split新增互不重合的D1（joint-0锁定）与D5（joint-4锁定），D3仍严格
test-only；shared h136与topology h136均用相同扩展数据、240 epochs/h5重训。
seed7 topology scaffold的primary为free +4.46%、object -10.97%、overall +4.08%；
接回independent object32后的四域为free +0.21%、object -36.44%、overall -0.64%。
第三、第五锁定域提供了一定robot元训练信号，但更强公平baseline同时显著提高object，
旧object32已不再占优。

Z33 精确参数匹配 compact bridge（2026-08-22）：保留Z32 topology robot，在同一
BT-DPWM内用`stop-gradient(mean(robot hidden)) + object state`驱动两层object head；
总参数338,102，与扩展shared baseline完全相同，object loss仍不能更新robot block。
seed7 primary达到free +5.11%、object -0.93%、overall +4.96%，但四域仅free
+1.07%、object -1.58%、overall +1.00%、1/4退化。逐域中D2 mixed为+12.56%，
D3 composition为+4.96%，D3 unseen为+0.23%，唯一主要失败是D4 mixed -13.74%。
因此compact bridge解决了容量公平与大部分object差距，但尚未形成跨拓扑+5%优势。

Z34 同结构低学习率校准反证（2026-08-22）：从Z32已选scaffold出发，仅在原训练split
以lr=1e-4追加40轮robot rollout校准，再以lr=1e-3重训同一compact object head。
robot训练loss由0.007286降至0.006260，但primary变为free -19.28%、object -0.23%、
overall -18.88%。这再次确认当前瓶颈不是欠优化；继续降低训练loss会破坏held-out
拓扑泛化。Z34冻结为NO-GO，seed17/27不启动。当前可继续的唯一机制方向是在训练域
内部预注册的拓扑稳健目标（如leave-one-topology-out worst-group selection），而非
增加epoch、按测试域调权或更换BT-DPWM。

Z35--Z37 拓扑稳健训练与正确robot-only校准（2026-08-22）：Z35以每轨迹loss按
topology聚合，固定优化`0.5*group mean+0.5*worst group`，validation使用同一准则；
尽管worst-group validation降至0.00855，primary仍free -21.53%、overall -21.07%。
审计随后发现Z34完整step校准时compact object head尚为随机初值，错误object rollout
污染下一步robot输入。Z36改用严格`step_robot`路径后，overall退化从-18.88%缩至
-4.43%，证明该实现错配真实存在；但40轮仍不泛化。Z37把初始checkpoint纳入候选，
每5轮按validation free-joint worst-topology选择，最终仍选epoch40且primary overall
-2.79%。因此修正训练路径和稳健早停能降低伤害，但已见拓扑validation仍不能选择
对未见D3有利的权重更新。

Z38--Z40 单模型安全路径与权重插值（2026-08-22）：Z38直接复制扩展公平baseline的
robot权重，将baseline训练中从未激活的mask/angle输入列显式清零，仅保留解析状态/
动作投影，再训练参数完全匹配的compact bridge。primary三项同时约+0.86%，但四域
free +0.23%、object -3.28%、overall +0.18%、2/4退化。Z39在相同318,378个robot
参数内，对projected-shared与Z32 topology权重预注册扫描alpha=[0,.25,.5,.75,1]，
只按validation free-joint worst-group选择；选择alpha=0，明确拒绝topology权重方向，
结果与Z38一致。Z40进一步直接复制并冻结baseline object head，得到primary及四域各项
约0.00%，验证参数映射、bridge形状和评估无隐藏偏差。解析投影是严格安全约束而不是
free-arm收益来源；达到+5%仍需要新的可迁移robot dynamics信息，不能由object容量、
更多epoch、已见拓扑group-DRO或权重插值产生。

Z41--Z44 公平短时system-ID（2026-08-22）：对每个冻结测试域另采与evaluation目标/
seed不重合的1条60-step calibration轨迹，shared与BT-DPWM均只估计同维输出校准量，
并用轨迹后半段从固定shrink集合选择。常数bias（Z41）四域free +2.02%、object
-2.66%、overall +1.92%、1/4退化，三个域均选择zero correction。逐状态delta
gain+bias（Z42，每模型同为28个部署系数）达到free +5.16%、overall +5.00%、
1/4退化，但object -4.33%，未过严格gate。robot/object分别选择shrink的Z43为free
+4.67%、object -1.92%、overall +4.53%。Z44将topology robot与baseline object head
直接组合，因hidden表示不兼容，零校准primary object -34.01%；block-affine后四域仍
仅free +4.71%、object -2.74%、overall +4.54%。公平few-shot产生robot信号，但未能
同时保持object正收益。

Z45--Z46 object选择与更多校准观测（2026-08-22）：Z45冻结Z32 robot，仅对compact
object head使用`0.5*group mean+0.5*worst topology`训练并按validation object rollout
选择；validation降至0.000829，但四域object -15.71%、overall +1.03%，再次证明weak
validation residual不能选择mixed test object泛化。Z46将公平block-affine预算提高到
3条×60-step（并修正cache key纳入trajectory count），结果反降至free +2.44%、object
-3.68%、overall +2.29%。因此当前one-step bias/affine system-ID与10-step rollout目标
不一致；不得以Z42 seed7接近+5%的单次结果升级主张，短时affine通道冻结为NO-GO。

Z47 rollout-aware公平system-ID（2026-08-22）：为排除one-step目标错配，shared与
BT-DPWM均冻结全部网络，只优化相同14 gain+14 bias；3条×60-step calibration中前2条
拟合、第3条每5步选择checkpoint，目标严格改为正式评估使用的10-step rollout终点，
并包含相同identity正则与系数边界。seed7四域结果为free -11.27%、object -6.20%、
overall -11.01%、3/4退化；D3 composition与unseen分别overall -16.48%和-23.06%。
validation在多个域持续选择最终step，但不能外推到disjoint evaluation trajectories，说明
阻塞不再是one-step/rollout目标差异，而是当前calibration激励缺少deployment代表性。

**恢复后的固定目标再次阻塞**：Z32--Z47依次加入额外锁定拓扑元训练、公平强baseline、
精确参数匹配bridge、group-DRO、正确robot-only rollout、validation早停、projected-shared
安全路径、权重插值、1/3-shot公平bias/affine与rollout-aware system-ID。连续证据均指向同一
条件：当前train/validation/calibration观测不能确定held-out mixed损伤上的robot correction
方向。继续在冻结四测试域上选择结构、步数或系数将构成test tuning。解除阻塞必须由项目
负责人批准一种新信息源（独立development mixed域、与部署相同的校准激励/传感观测，或
重新定义few-shot协议并重训同预算baseline）；在此之前不得宣称达到三seed×四域+5% gate。

---

## 20. 当前 Gate 决策与下一 owner

**当前阶段**：G0 已通过；G1 原始 DFWM No-Go；G1 robust zero-shot Pivot
已完成五 seed 最小预测机制验证并通过阶段交付；G2 强基线、结构化反证、选择性预测、
K0--K2 与 Gate L 已完成并冻结。

**当前 gate**：G2 结构化完整 world-model 主线停止。普通 ensemble 与 selective
prediction 证据成立，FT-GWM K1 仅保留为 provisional constraint-preserving
joint-dynamics 结果。Guarded MPC 的统计稳定收益未成立，正式 G3 尚未批准。

**下一执行 owner 与顺序**：

1. artifact owner：提交并冻结 G2 config、split、实现、测试、最终结果和审计报告；
2. 论文 owner：按最终 synthesis 重写摘要、贡献、方法边界和实验结论；
3. 真机 owner：若项目负责人批准 G3，先执行 intact/D3 安全与接口 smoke；
4. uncertainty owner：按部署固定 horizon 重新校准 rejection gate，不复用 mixed-depth 阈值；
5. 项目负责人：决定收缩后的 ensemble/selective-prediction 主张是否值得投入 G3 正式统计。

**下一批必须回答的问题**：

- 收缩后的 ensemble/selective-prediction 结果是否足以形成可信的 ICRA 投稿故事？
- 固定部署 horizon 下 disagreement 的排序能力与拒绝阈值是否稳定？
- 若进入 G3，两个故障 condition 的最低证据包能否在安全和时间预算内完成？
- 若真机控制收益仍不稳定，论文是否明确定位为 dynamics prediction/uncertainty？

---

## 21. V6 完成标准

本计划本身完成不等于项目完成。项目完成的最终定义是：

- 研究问题与方法无内部矛盾；
- G0–G4 留有可审计 artifact；
- 结果支持的主张与论文一致；
- 失败结果和适用边界被披露；
- 真实系统安全、统计和成本透明；
- 官方会议规则在提交前重新核验；
- 不依赖预期数字、不可复现脚本或单次展示。

---

## 22. BT-DPWM 真机执行状态（2026-08-22）

Z69 已修正旧 BT checkpoint 将 shared 训练中从未激活的 topology 输入列随机带入推理的
问题；三 seed 的 K0 overall 相对公平 shared 仅为 -0.33%、-0.71%、-1.49%。Z70 使用
完全相同的 adapter 参数量、真实 transition 预算和优化协议，BT 相对 shared 在 K10/K25
分别达到 +0.04%/+0.03%，K50 为 -0.19%；BT 自身增益随预算达到 +7.24%，全部域无
负增益或安全回退。该结果支持“解析投影约束下的安全 few-shot physical-context 适配”，
但不支持声称 object expert 已全面超过 shared，后者仍是公开风险边界。

真机接口已固定为 ST3215 J1--J5（ID 1--5）、位置/速度/电流/温度与眼在手视觉，不要求
力传感器。D2/D3 各三次、K=5/10/25/50 的采集器默认 dry-run，只有显式 `--execute`、
安全确认字符串和新鲜视觉位姿同时满足才允许运动；锁定维动作严格为零，并具有电流、温度、
锁定漂移和视觉超时的 torque-off 路径。

眼在手视觉采用固定桌面 reference ArUco 与 object ArUco 的同帧相对定位，使相机随手臂
移动时仍输出桌面参考系下的平面位置和速度。程序要求真实相机内参、畸变系数及实测 marker
尺寸，缺任一项即拒绝运行，不用仿真标签替代真实观测。当前主机预检未发现串口设备，项目
虚拟环境未安装 OpenCV，仓库也没有相机标定，因此目前只有 dry-run 与接口测试，尚无 D2/D3
真机证据。下一不可跳过步骤是连接机械臂与相机、完成相机标定并实测 marker 尺寸，然后先做
无运动视觉 smoke，再做单个 D2 K5 低幅安全 smoke；通过后才能展开预注册矩阵并生成最终报告。

相机内参标定与 readiness gate 已实现：标定要求不少于10张不同视角、相同分辨率的棋盘格
图像，并冻结重投影 RMS ≤1.0 px；未达标不生成 calibrated 配置。只读 readiness 审计同时
检查硬件安全配置、舵机串口、OpenCV ArUco、合格内参与实时视觉流，并将是否允许低幅 smoke
写成机器可读 JSON。2026-08-22 当前审计为 NO-GO（仅硬件配置通过；串口、OpenCV、标定、
实时流未通过），且确认未执行运动。该 gate 必须先转为全绿，不能用 synthetic pose 绕过。

真实 transition budget 协议进一步冻结为 nested-prefix：每个 topology/repetition 只采集一条
K50 安全激励轨迹，K=0/5/10/25/50 均从同一条轨迹依次取前缀。执行模式拒绝单独采 K5/K10/
K25，防止把不同初始状态或重复运动混入样本效率横轴；因此 D2/D3×3 的物理采集总量为6条
K50轨迹，而不是24条相互独立的预算轨迹。

项目依赖现以 `pip install -e .[real]` 显式安装 OpenCV Contrib/ArUco，并提供只读交互式
棋盘格采图工具（仅检测成功时保存，至少10张，建议20张且覆盖不同位置、倾角、距离和画面
边缘）。当前主机已安装 OpenCV Contrib 5.0.0，ArUco API 通过，并检测到 camera index 0；
readiness 中视觉软件项已转绿，尚需实体棋盘格图像与标记实测尺寸才能生成真实标定。

---

## 23. Z71--Z75 五 seed G2 补证据与安全机制审计（2026-08-23）

为满足原G2五seed要求，在查看新结果前预注册 seed37/47，与7/17/27组成固定集合；V0、
Z32、Z69、shared/BT adapter、Z65 encoder和Z70 evaluation均使用相同数据、初始化规则与
优化预算。旧双专家 V0 在37/47均NO-GO；旧Z32的free回归分别为-18.31%/-52.59%。Z69
只清零shared训练中从未激活的topology输入列并保留解析投影，将其恢复为+0.09%/-1.33%，
再次复现根因修复。

原Z70扩到五seed后，K50 BT-own均值+5.72%，seed-bootstrap 95% CI [+3.55%,+8.06%]，
但seed47的D3 composition/unseen出现-9.96%/-5.46% evaluation负迁移。审计发现三处实现与
安全主张不一致：uncertainty proposal第一次接受绕过统一hysteresis；接受后不再保留z=0
候选；单个最新support-validation窗口会遗忘先前预算的验证证据。Z72--Z75在同一BT-DPWM
机制内依次修复为首次/替换统一门槛、永久z=0候选、D3/D4至少15个fit transitions、
`context_mean_std<=0.30`及K25/K50嵌套support窗口的1%非回归约束。

Z75五seed development结果为：BT-own K=0/5/10/25/50均值
`0/0.504/0.504/3.357/3.357%`，全部seed/domain/budget非负、均值严格不降、constraint
violation为0；K25/K50 seed-bootstrap 95% CI约为[+1.36%,+5.61%]。公平shared曲线为
`0/0.548/0.548/3.377/3.377%`，故BT在K25低0.020个百分点，预注册“严格大于shared”
符号门未通过；K50 BT相对shared为-0.778%，仍在预设-1%工程容忍内，但不得宣称性能领先。
Z71失败结果及Z72--Z74开发轨迹全部保留，不能用Z75覆盖。由于Z75由五seed安全审计产生，
论文级确认还需要未参与改动的独立confirmation seed；在此之前将其标为development pass on
safety/monotonicity，而不是最终G2 Go。

---

## 24. Z76 独立确认结论（2026-08-23）

在任何seed57/67 checkpoint产生前，独立冻结两seed、四域、K=0/5/10/25/50及1个百分点
paired-equivalence门槛；Z69基础机制、Z70公平adapter、Z65 uncertainty encoder和Z75嵌套
support安全规则均未根据确认结果修改。两seed均完整保留V0、Z32、Z69和最终Z75链路。

确认集再次暴露旧双专家不稳定：seed57 V0 free +6.14%但object -2.82%；seed67 V0 free
-25.98%（相对compute-matched为-32.28%）。seed67 Z32将free回退缩至-16.06%，Z69仅通过
清零未训练topology输入列并保留解析投影，将free/overall恢复到+0.47%/+0.27%；seed57
Z69为free/overall +2.21%/+2.05%，但object仍-20.42%。因此这些Y0结果继续作为NO-GO和
根因恢复证据，不替代最终few-shot结果。

Z76最终BT-own K=0/5/10/25/50均值为`0/3.880/3.880/8.065/8.065%`，shared为
`0/3.933/3.933/6.440/8.686%`。全部seed/domain/budget的BT-own gain非负，均值单调，
constraint violation为0；K50 BT-own两seedbootstrap区间为[+3.24%,+12.89%]。这使
“可逆、安全、能产生有用改善”的窄主张获得独立确认。

更强的paired sample-efficiency gate未通过：K25 BT-minus-shared CI下界为-0.051个百分点，
在1pp等效界内；K50下界为-1.191pp，比-1pp门槛差0.191pp。尽管K50 BT相对shared均值
为+0.092%，仍必须按配对下界判NO-GO，不能声称独立确认下等效或领先。当前仍处G2补证据，
不是完整G2 Go，更未进入G3。下一步冻结方法，完成BT机制消融、robustness、uncertainty
calibration、compute/failure ledger和统一论文表；不得在seed57/67上调Z75阈值。
