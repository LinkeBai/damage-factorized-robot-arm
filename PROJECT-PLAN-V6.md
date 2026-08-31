# Project Plan V6 — Verified SI-IPWM + Pending Few-Shot Fault Recovery

## 2026-08-31严格论文闭环阶段（当前最新）

- 已在生成数据前登记 `reports/post-freeze-d3-confirmation-protocol-20260831.md`，
  随后一次性生成 D3、seed `91031` 的新候选查询集：200组、每组128条独立
  50步轨迹，共25,600行，零重复；协议审计通过，SHA-256为
  `43a00365caf59e504ef7b730fc9d91bc7bfd0d9efce79899a7b9d725072e2702`。
  三个冻结checkpoint只在该集合上正式评测一次。由于D3在历史探索中已被看过，
  该证据准确称为“post-freeze untouched candidate-query confirmation”，不是
  pristine unseen-domain confirmation。
- D3确认结果没有形成预声明的巨大优势：同容量global residual相对nominal的
  top-1 regret平均降低`9.77%`，仅2/3种子为正；终点误差平均降低`2.00%`，
  仅2/3为正；成功率平均提高`2.17 pp`且3/3为正。接触响应RMSE平均恶化
  `263.63%`，Spearman平均下降`0.0148`。因此它只确认了小幅成功率方向，未过
  预注册的10%/3-of-3中等优势门，更未过20%强优势门。
- selective IPWM在D3上相对global residual的regret为`-3.82%`、终点为
  `-0.50%`、成功率为`-0.83 pp`，三项都只有1/3种子占优；选择性结构归因继续
  `NO-GO`。不得把seed27单独的regret `+16.59%`写成总体贡献。
- 当前可写的“大幅”指标只有结构正确性：解析投影把最大锁角漂移从平均
  `6.63°`降为精确`0`（约束违例100%消除）。任务相关最大稳定指标仍是D2/D4
  开发集global residual相对nominal的regret `19.76%`（3/3）；D3确认表明该幅度
  不稳定外推，论文必须将其限定为开发证据。
- 真机主比较已随证据归因纠正为`nominal`对`global_matched`；两者是当前唯一
  具有稳定仿真控制信号的公平比较。`si_ipwm`作为同一pair内的第三方法仅用于
  选择性结构归因，不能预设为赢家。真机分析器现支持显式reference/candidate、
  配对位置一致性、重复行拦截，以及reach/contact/success/endpoint四类配对统计；
  相关回归测试通过。
- 旧合同审计曾错误保留“正式结果0/8”的实验前状态。现已改为从权威JSON动态
  验证，仿真合同为`9/9`同协议单元完整（额外显式包含oracle），状态为
  `SIMULATION_EVIDENCE_COMPLETE_REAL_ROBOT_PENDING`。三seed realized-cost oracle
  的平均终点误差为`0.03758 m`，nominal为`0.04675 m`，即仍有`9.17 mm`
  或`19.63%`终点headroom（3/3为正）；oracle仅用于上界诊断，不是可部署基线。
- 次要夹取不再停留在计划文字：已增加固定预抓取短抬升模板、严格文件/数值/
  abort校验和按intact/D2/D3汇总的分析器。该面板冻结为每条件最多5次、保持3秒，
  只能支持reach/closure/retention feasibility，禁止包装为learned grasping或方法
  性能比较。Push仍拥有真机时间和论文主证据的最高优先级。
- ICRA desk审计已机器化并通过：当前PDF 7页、US Letter、匿名作者可见、身份相关
  metadata为空、未出现配置的本地/GitHub身份字符串、21/21字体均为嵌入Type 1；
  第1/4/7页渲染抽查无裁切或重叠。真机图加入后必须重新运行该审计。
- 真机Push主排程已用seed `20260901`预生成并冻结：25个paired blocks、50次
  trials（intact 5对，D2/D3各10对），pair内`nominal/global_matched`顺序随机且
  共享位置；SHA-256为`79139bca3b61866643e00ef35d724cdd4185fb14a8f115faa942635f27f4510d`。
  若现场必须缩短，只能在trial 1前记录dated deviation，禁止看结果后删改排程。
- 真机统计已补齐反偏差规则：任何一方abort/缺失的pair均进入
  `incomplete_or_aborted_pairs`，不得静默从分母删除；D2和D3必须分别达到10个
  完整pair且原始文件检查通过才标为formal。汇总保留逐pair差值和逐故障bootstrap
  CI，并可直接生成论文PDF图与LaTeX表，避免手抄数字。合成数据仅用于版式测试，
  已删除且不会进入结果目录。
- 复现环境已冻结并机器审计通过：Python `3.12.10`、NumPy `2.5.2`、MuJoCo/
  MuJoCo Warp `3.12.0`、PyTorch `2.11.0+cu128`、SciPy `1.18.1`等精确版本，
  同时检查Poppler/LaTeX命令可用性；输出已去除本机绝对路径，不能泄露匿名身份。

- goal现采用可审计闭环而非开放式试错：`同协议开发消融→六阶段定位首个
  失败层→一次只做一个预声明调整→重跑全部开发seed→冻结checkpoint→一次性
  确认→更新主张与论文`。循环规则已写入主协议的`iteration_loop`。D3不得用于
  权重搜索或反复确认；确认失败只能回到未使用D3的开发证据、收缩主张，或在
  明确修订协议后建立新的确认资产，不能靠挑seed闭合。
- 严格研究 goal 已于本轮重新建立并锁定：不换问题、不换主模型、不把探索性
  validation 增益当作确认结果。逐结果的模型身份、数据、seed、指标与可主张
  边界统一记录在 `reports/primary-result-provenance-ledger-20260831.md`；该台账
  是主稿和后续表格的强制引用源。
- 重新运行四组核心专项测试，结果为 `54 passed`。这证明成对候选加载、128
  候选协议、解析投影开关、全局同容量头和选择性 rollout 的实现仍可执行，
  不证明任何方法具有性能优势。
- 必须纠正此前最容易混淆的正结果：D3上终点误差降低`8.93%`、成功率提高
  `10.33`个百分点、top-1 regret降低`26.97%`且3/3 rollout seeds方向一致，
  来自另一个直接预测14维状态的紧凑序列模型。它有解析投影和decision loss，
  但没有共享carrier与选择性发布，因此只能支持“硬约束＋决策相关序列训练”，
  **不得归因给选择性IPWM**。D3虽未参与最终拟合与epoch选择，但此前已被多次
  检视，只能称“held out from final fitting with fresh evaluation seeds”，不能称
  pristine never-seen confirmation。
- 权威SI-IPWM的weight-10、40-epoch开发训练覆盖320/320训练组，32候选validation
  上Spearman约从`0.0164`升至`0.0636`，top-1 regret约下降`4.2%`。随后发现此前
  独立400组×128候选重评使用了错误CLI：`--initialize-candidate-model`只加载
  `robot_`前缀参数（317,834/337,834），没有加载物体头和选择性修正头，却以
  epoch-32完整模型名义报告完全并列。该No-Go现已撤销为**INVALID checkpoint
  reconstruction**。新增`--initialize-candidate-full-model`实行全键、全形状严格
  匹配；smoke已确认337,834/337,834参数加载。正式V2重评正在生成，完成前不得
  宣布选择性机制Go或No-Go。
- 六阶段首轮还发现候选生成器只记录`tool_geom`接触，漏掉实际推块使用的
  `pusher_geom`。V2生成器改为同时监测两者，并记录每个10步segment内连续
  最小MuJoCo几何距离。完整V2 seed7保持动作、状态、终点代价和成功标签逐元素
  不变，只新增3,835个漏标接触；400组×128协议审计通过，SHA-256为
  `70e9ed782bb24508776e93f44ca66fd0e4a8abdbe0870f7f7396119d22443039`。
- 严格完整checkpoint的seed7正式结果产生真实但未过门的正信号：相对carrier，
  接触响应RMSE改善约`4.35%`，Spearman绝对提高`0.03656`，top-1 regret降低
  约`10.56%`，终点误差降低约`1.99%`，成功率提高`0.25`个百分点。full-state与
  selective逐项相同，故该信号不能归因于选择性发布；selective六阶段耗时约
  `333 s`，full-state约`200 s`。
- seed17按相同训练、严格加载和独立400×128 V2协议复核后方向不一致：相对
  carrier，响应RMSE恶化`32.11%`、Spearman仅`+0.00418`、regret恶化`1.08%`、
  终点恶化`0.19%`、成功率下降`1.25`个百分点。weight-10当前为1/2正向，不能
  宣称稳定贡献。seed27前置机器人checkpoint已补齐，第三种子decision训练正在
  执行，完成前不做2/3结论。
- seed27严格结果现已完成：Spearman `+0.01996`、regret改善`3.83%`、终点
  改善`0.584%`、成功率`+0.25 pp`，但响应RMSE恶化`26.46%`。机器汇总
  `results/final/primary-strict-development-3seed-summary.json`给出完整三seed结论：
  Spearman 3/3正向、均值`+0.02023`；regret 2/3改善、均值`4.44%`；终点2/3
  改善、均值仅`0.796%`；成功率均值`-0.25 pp`；响应RMSE仅1/3改善、均值
  恶化`18.07%`。方向门通过但幅度门0/3通过，full-state与selective在3/3
  seed逐项相同，故正式判定为`DIRECTIONAL_SIGNAL_MAGNITUDE_AND_ATTRIBUTION_NO_GO`。
- CPU/CUDA设备等价性已在同一seed27、2组×128候选smoke上验证：四方法的
  Spearman、Kendall、regret、终点误差和成功率最大绝对差均为`0.0`。CUDA正式
  六阶段耗时约为shared `22.8 s`、carrier `66.1 s`、full-state `62.7 s`、
  selective `113.7 s`；CUDA适合大批评测，但小批50步训练受Python循环和显存
  竞争影响反而更慢。
- 同一机器汇总现同时报告nominal/shared总体对照与carrier增量对照。完整方法
  相对nominal的top-1 regret三seed均改善，平均`18.28%`、范围
  `[9.87%,29.17%]`；Spearman三seed均提高，平均绝对`+0.03784`；终点误差
  三seed均降低，平均`3.73%`；成功率三seed均提高，平均`+1.17 pp`。这是目前
  最大且稳定的任务相关优势。与此同时接触响应RMSE相对nominal平均恶化约
  `247.90%`，形成“响应RMSE显著变差但动作regret与实际结果改善”的核心诊断
  证据。必须同时披露carrier增量仅为regret `4.44%`、终点`0.80%`且选择性发布
  无独立效应；不能把完整管线相对nominal的`18.28%`全归因于选择性机制。
- 无decision-loss三seed同协议消融已完成，机器结果为
  `results/final/primary-decision-loss-ablation-3seed.json`。weight-10相对weight-0
  的regret 2/3改善、均值`5.98%`，终点2/3改善、均值`1.40%`；但Spearman仅
  1/3改善且均值`-0.0108`，成功率均值`-0.50 pp`，响应RMSE 3/3恶化、平均
  恶化`356.44%`。正式结论为
  `DECISION_LOSS_REGRET_ENDPOINT_DIRECTIONAL_ONLY_RESPONSE_SUCCESS_NO_GO`，不能
  将完整方法相对nominal的18.28%优势归因给decision loss。
- weight-0故障感知结构相对nominal表现更均衡：响应RMSE 3/3改善、平均
  `24.84%`，成功率3/3提高、平均`+1.67 pp`，regret平均改善`10.31%`且2/3
  正向，终点平均改善`2.29%`且2/3正向。下一允许调整不是新模型或继续放大
  decision weight，而是在weight-0结构上使用受响应稳定门约束的轻量decision
  目标；进入该单变量修复前仍需先完成全局同容量与无投影归因。
- 计算环境审计：普通`.venv`为CPU-only PyTorch 2.13；`.venv-cuda`为PyTorch
  2.11+cu128且RTX 4060 8GB可用，MuJoCo Warp已安装。seed27 decision训练已切
  CUDA，但物理候选生成仍是标准MuJoCo串行代码，不能宣称Warp并行；当前GPU还
  与桌面/游戏进程竞争（审计时显存约6.26/8GB、利用率32%），效率结果需注明。
- 已确认桌面Git仓库和GitHub为权威项目；远端已快进同步。此前报告的源码缺失仅发生在另一个无版本工作副本，原始IPWM源码在权威仓库完整。
- `config/experiment/icra_2027_primary_5dof_recovery_v1.yaml`已经冻结五方法、三消融和D2/D4→D3协议，但尚无统一执行入口或同协议结果，不能把配置清单视为实验完成。
- 权威SI-IPWM实现包含解析投影、carrier、双私有rollout、选择性物体发布与已接入
  的成对soft-regret训练。另一工作副本的128候选序列模型虽同样含decision loss，
  但并非完整SI-IPWM，二者仍不得互相改名或拼表。
- 已新增机器可读合同覆盖审计、原始臂真机Push配对模板和分析脚本。下一实现只允许在现有SI-IPWM训练链增加成对候选损失与统一组件开关；选择与调参仅使用D2/D4，D3保持确认。
- 完整审计见`reports/strict-primary-contract-audit-20260831.md`。决定性八行消融未完成前，评分仍不得宣布4.0+/5。
- 成对soft-regret损失、严格分组加载器和同checkpoint的carrier/full-state/selective评估现已实现并通过专项测试。首个D2/D4 smoke产生非零梯度，但验证选择epoch 0；三方法排序相同，Spearman `-0.0362`、top-1 regret `0.01238`，判定为管线通过、性能No-Go。该结果不进入论文主表。
- “无解析投影”现为同架构、同权重的显式消融开关，专项测试证明关闭后锁定坐标可以漂移。全局同容量修正也已实现：与选择性头共享12维输入但可发布到全部14维，冻结rank下整模型仅多8个参数（远小于0.1%），硬投影仍保留。48项相关测试通过。两项D2/D4四组、32候选接线smoke均为No-Go：无投影overall `-0.14%`、Spearman `-0.0415`；全局修正object `-2.55%`、overall `+6.55%`、Spearman `-0.0362`，二者top-1 regret均约`0.01238`。因此八个冻结消融单元实现已达8/8，但正式128候选、多seed和D3同协议结果仍为0/8，当前不能报告性能归因。
- 已生成并审计seed 7的完整D2/D4开发候选集：400组、每组128条独立动作序列、50步、零重复、51,200行，SHA-256为`d587bd32de45ffe76ccee6c25adfcf98099a35a934169b801c08d14e64425180`。现有epoch-0-selected checkpoint的全量诊断显示carrier与full-state/selective完全并列：Spearman `0.02116`、Kendall `0.00278`、top-1 regret `0.00906`；oracle cost `0.03570`，所选cost `0.04476`。这将“候选不足”排除为当前主要原因，当前No-Go定位为决策修正没有被验证选择；单seed诊断不填正式消融表。

## 当前权威执行摘要（截至 2026-08-30）

> **阅读规则。** 本文件是当前唯一权威计划。已经失效的逐轮计划、旧时间表和
> 重复主张已从正文移除，由 Git 历史与 `reports/` 下的独立审计报告留档。
> 近几天真实尝试、纠错与 No-Go 仅以压缩证据账本保留。旧标题中的
> `zero-shot` 已不再是论文核心主张。

### 1. 三十秒结论

近几天已经完成大量机制、基线、仿真资产、复现和证据审计工作，但当前论文
仍不是一篇可客观评分为 4+/5 的 ICRA 稿件。工作量充分，筛选严格；真正缺少的
不是又一个组件，而是一个能在强基线之上稳定改善动作选择和闭环任务结果的
核心机制。

当前已经成立的结果只有：

1. 已诊断单关节锁定下，解析投影严格保证
   `q_j=qbar_j`、`qdot_j=0`、`a_j=0`；
2. 双私有 rollout 和选择性发布保证物体适配分支不能改写 carrier 发布的
   自由关节与推子状态；拒绝干预时精确回退；
3. 旧简化模型中，选择性 IPWM 在审计的 object-RMSE 单元保留预测收益；
4. 普通 ensemble 是有效预测基线，代码、原始 JSON、冻结配置、失败记录和
   测试链已经形成可复核资产。

当前没有成立的结果包括：

- 原始 5-DoF 或校准 GenkiArm 上的稳定任务性能优势；
- 预测改善稳定转化为 action ranking、top-1 regret 或闭环成功率改善；
- 跨机械臂 object/contact 动力学泛化；
- 方法级 Grasp、视觉闭环或真机收益；
- SFET/运输算子在机器人任务上的极少样本恢复优势。

最新关键纠错是：此前名为 `oracle_ik` 的方法没有使用未来状态或隐藏真值，
它是可部署的 **fault-aware constrained IK 强基线**。该基线在原始臂开发屏幕上
D2 为 `100/100/100%`，D3 为 `80/100/100%`，说明“已知锁定后的简单直线
Push”主要先是运动学可行性问题。后续所有方法必须共享该基线建立接触，不能
再通过比较弱 hard-mask 获得虚假的巨大提升。

按当前 ICRA/CCFA 证据合同，综合评分仍冻结为 **3.2--3.4/5**。No-Go 提高了
记录可信度，但本身不增加新颖性、证据或意义分。

### 2. 现实压力和已投入工作量

本节用于项目管理和向学长解释执行强度，不进入论文正文，也不用于降低统计
或证据标准。

- 截至本次冻结，距离投稿约 16 天；学长按约每两天检查一次进度；原始臂真机
  节点约在 48 小时内。
- 2026-08-28 至 2026-08-30 的工作区盘点包含 `38` 份新近报告、`46` 个新近
  脚本、`73` 个结果文件和 `424` 个 run 产物，run 目录约 `133 MB`。这些数量
  只证明持续实现、运行和审计，不等同于独立实验样本，也不能作为论文贡献。
- 本次 GitHub 收口纳入 8 月 30 日下午之后的 SFET、复现基线、新报告、补丁、
  本计划与小型机器可读结果；本地大体积 runs、外部完整仓库和模型缓存仍通过
  manifest/命令索引复现，不能假装已经随 Git 提交。
- 测试快照包括 solver/约束链 `259 passed`、Genki 原生训练改动 `16/16`
  回归通过、实际模型与相机资产 `9 passed`、SFET 结构与方向性工具 `6 passed`
  以及状态隔离/投影的专项复核。各套测试存在重叠，不把它们相加成一个夸大的
  测试总数。
- 多次 No-Go 不是没有推进，而是预注册门槛后主动停止了不能归因、不能闭环或
  被强解析基线覆盖的路线，避免继续消耗剩余时间。当前压力管理原则是：减少
  并行造模块，优先形成一条完整证据链；必要时先交可信的边界，不交无法通过
  消融的漂亮数字。

### 3. 近几天工作量与证据账本

| 工作包 | 已完成规模与产物 | 结果 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| 旧简化臂五种子预测审计 | D2/D3、多个物理域、H10/H25/H50；原始表和 bootstrap 保留 | K25 IPWM 相对 matched shared 的 object-RMSE 跨种子均值约改善 `8.99%--26.58%`；审计单元方向一致 | 简化臂中选择性物体预测有信号 | 原始臂、真机或闭环优势 |
| ensemble 归因审计 | 五种子、参数匹配 single/ordinary/structured | ordinary ensemble 相对 single `+30.74%`，95% CI `[15.06%,42.62%]`；structured 相对 ordinary 仅 `+2.47%`，CI 跨零 | ensemble 是强基线 | 把 30.74% 归因于 IPWM 结构 |
| free-joint 失败定位与选择性发布 | 三种子、27 个审计单元、H10/H25/H50 | routed object 27/27 为正，均值 `+17.04%`、bootstrap CI `[6.92%,26.34%]`；锁定违例为零，选择性发布不改变 carrier 自由状态；闭环仅 1/3 改善 | 状态隔离机制与失败位置 | 任务安全或控制占优 |
| physical-support router | 42 个 physics×horizon 单元 | object 31/42 非负、42/42 在 2% 回退边界内；但最差 free-state 回退 `24.456%` | 支持域/拒绝机制可限制 object 回退 | free-state 安全门或完整方法通过 |
| 动作排序和闭环诊断 | CEM、候选重排、接触 ranker、terminal ranker、one-shot MPC 多轮 Gate | 多个 ranker 在离线 Spearman 上出现正信号，但闭环仍差于 carrier；最终均 No-Go | 确认 RMSE 与控制目标错配 | 用离线 ranker 数字代替闭环证据 |
| solver-native/约束响应 | 解析分解、前缀回放、接触响应、held-out joint，三种子 | 实现和单元测试通过；deployable held-out lock 仅 1/3 正向 | 解析响应可实现、反力包含信息 | 稳定的未见锁定传播算子 |
| GenkiArm 冻结迁移与确认 | seeds `107/117/127`，D3 物理 OOD，多 horizon | routed SI-IPWM 为 `+0.604/-0.784/+0.718%`，均值 `+0.179%`，2/3 为正且区间跨零；锁定违例与自由状态变化为零 | 实际模型上复现状态隔离 | 性能优越性 |
| variable-DoF 与双臂机器人转移 | 同一 42,626 参数共享模型；Genki 5-DoF + Panda 7-DoF；held-out middle joint | robot-transition Gate 为 2/3 小通过，pooled 改善 `67.45/59.76/20.77%`，但 seed17 的 Genki 回归 | 多 DoF 接口和局部结构共享可行 | 未见机械臂、对象或接触泛化 |
| 双臂 object/contact Gate | 每臂 80 个接触前缀、三种锁定，共 480 条严格同状态反事实记录 | structured 相对 flat 为 `-28.00/-63.58/-13.18%`，0/3 | 公平排除了当前共享接触头 | 跨臂 object/contact 传播 |
| H10 contact action-effect Gate | 两臂各 80 prefixes、每 prefix 6 actions、3 locks，共 `2,880` 条严格同状态 branches | 参数匹配结构模型 0/3；Spearman 和 regret 恶化 | 候选动作信号存在且协议可复用 | 低秩传播算子有效 |
| 多步可观测性 | H1/H5/H10，同状态候选分支 | H1 动作差过小；H10 候选 XY 范围约 2.4 mm，但接触保持仅 Genki 60.7%、Panda 45.6% | 最终实验必须使用多步/完整轨迹 | H10 信号自动带来成功率提升 |
| hybrid contact/mode factorization | contact survival、slip mode、累计响应联合 Gate，三种子 | 0/3 | 接触模式可测但当前机制不足 | “混合模式”本身是创新或已成功 |
| active-contact history | 两臂 `14,400` rows；80 prefixes/arm × 5 physics × 3 locks × 6 actions；K8 probe + H10 branch | 原报告 pooled RMSE 有信号，但跨臂 0/3；后续审计发现旧排序分组遗漏 physics profile，旧 `+0.384--+0.556` Spearman 不得使用 | 历史包含 embodiment-dependent 局部响应信息 | 部署内排序改善或统一跨臂 decoder |
| 操作空间无量纲化 | mobility、有效质量、能量与接触坐标归一化，三种子 | pooled RMSE `+23.65/+16.05/+19.28%`，但联合控制 Gate 0/3 | 归一化缓解部分尺度不对称 | 动作排序和闭环创新 |
| 形式性质 | 硬投影、双私有 rollout、选择性发布、guard 风险分解 | 专项测试通过；给出 exact feasibility、carrier-relative non-interference、conditional fallback bound | 可证明的结构贡献 | 相对真值安全、性能或普适鲁棒性 |
| 实际模型和视觉资产 | 修复 Genki/Panda 右相机朝向及 Panda task-home；两臂各两台固定 eye-to-hand | 九种外参扰动 × 两臂共 18/18 条件保持双相机可见 | 相机布置和任务可观测性可行 | 学习视觉、双目融合或视觉鲁棒控制 |
| Panda Grasp 可行性 | 无 weld 的 scripted IK，五个方块扰动 | 5/5 成功，平均抬升 0.142 m | Grasp 任务和物理接触模型可运行 | IPWM 改善 Grasp |
| 近邻项目复现审计 | TD-MPC2、LeWM、DINO-WM、DyWA、fault-locomotion、IROS failure-NPM 等 | TD-MPC2 官方短程复现可运行；LeWM checkpoint/PushT/CEM 链可运行；部分项目因无代码、数据过大或 Windows/IsaacGym 不兼容仅完成审计 | 大幅提升通常来自控制对齐目标、强数据覆盖和完整训练栈 | 把别人的报告数字当作本地复现结果 |
| TD-MPC2 原始臂迁移 | 1.2k/5k-step smoke、接触覆盖审计、guided replay | 随机 15k steps 为 0 接触；修复方向 waypoint 后 20 episodes 达到 100% 接触、75% teacher success；guided TD-MPC2 5k 仍 0% policy success | 找到数据覆盖与控制接口问题 | TD-MPC2 或 IPWM 的最终优劣 |
| fault-aware/unaware 硬锁定筛选 | D2/D3/D4，各 3 个开发屏幕、每格 10 episodes | constrained IK 相对 unaware：D2 提高 70--80pp，D3/D4 提高 100pp；D3 主要不接触，D4 主要接触后推错/过冲 | 将故障问题拆成预接触可达性与接触后效果适配 | learned-model 优势或独立统计结论 |
| BC/数据聚合诊断 | 40 条 directional demos、5,101 transitions、20 unseen targets | BC 100% 建立接触，成功率 25%，终点误差 72.1 mm | 接口可学习，主要问题转向接触后分布偏移 | BC 已经是强任务基线 |
| healthy DAgger 控制对齐诊断 | 40 条初始 demos + 3 轮各 10 条聚合轨迹；3 seeds；固定 20-target evaluation | 第三轮后成功率为 `100/85/95%`，平均终点误差约 `9.46/11.40/10.34 mm` | 原始臂环境、观测和动作接口可被控制对齐训练解决 | 关节锁定恢复或世界模型贡献；该组为 healthy sanity |
| D3 五轨迹 DAgger | 5 条故障轨迹总预算、3 seeds、每种子 20 个未见目标 | 成功率 `55/25/75%`，均值 `51.7%`，约 588--618 transitions/seed | 少量完整轨迹可产生部分恢复，必须作为强数据聚合基线 | 稳定三试验优势或结构创新 |
| fault few-shot BC/HCAR 基线 | D3/D4 × 1/3/5-shot × 3 seeds；每格 20 个未见目标，共 720 evaluation episodes | 1-shot 几乎失败；D3 3-shot BC/HCAR 均值 `46.7/50.0%`，D4 为 `83.3/53.3%`；5-shot 结果非单调且两者互有胜负 | 少样本强基线和数据量边界已经可运行 | HCAR 稳定占优、IPWM 必要或独立统计结论 |
| LeWM PushT 本地复现 | 官方 18.0M 参数 checkpoint、CEM 路径；两个 goal offset，各 10 episodes | offset25：LeWM 4/10、random 1/10，`p=0.25`；offset50：1/10 对 0/10 | 上游模型、checkpoint 和规划链可运行 | 复现论文整体优势或直接迁移到关节锁定 |
| SFET 候选机制 | masked Broyden、局部任务响应运输、结构与工具测试 | 合成 2×5 响应中 3-shot 平均相对 nominal 降低 41.4%，仅 2/3 过 20%；5-shot ridge 更强 | 极少样本名义先验有局部信号 | MuJoCo Push、闭环或真机恢复 |
| SFET 原始臂载体筛选 | D2/D3 × 3 seeds × hard-mask/transport/constrained IK × 5 episodes，共 90 个开发 episodes | D2 transport 与 hard-mask 同为 `20/0/20%`；D3 二者均 0%；constrained IK 为 D2 `100/100/100%`、D3 `80/100/100%` | 单步运输不能代替整条故障可行路径；强基线暴露问题定义缺陷 | SFET 是核心正结果 |
| 晚间三点/割线/双线性上界诊断 | 修正 ranking group 为 robot/profile/prefix/lock；三种子、Genki/Panda | probe 可大幅降低部分 response RMSE，但 ranking/regret 不稳定；三点仿射运输即使使用真实 intact 未来作完美先验，排序仍不改善 | “拟合更准但不会选动作”是主阻塞 | 继续用另一个效果运输名称包装成功 |
| Genki 数据 provenance 修复 | 发现 CPU collector 曾静默回退 `arm_push.xml`；隔离无效缓存，把 XML provenance 加入 cache key/调用链并重启 fresh pipeline | 早期混合 XML 数据作废；fresh seeds 107/117/127 重新确认 | 数据来源和证据链完整性 | 把被污染旧结果继续算作 Genki 证据 |

#### 3.1 近邻项目复现状态专表

“复现”按完成程度分级，不能把下载代码、通过编译、跑通一个 episode 和复现论文
统计结果混为一谈。

| 项目 | 本地完成内容 | 当前等级 | 对本项目的直接启示/下一步 |
|---|---|---|---|
| TD-MPC2（ICLR 2024） | 固定上游 commit；Windows/MuJoCo 依赖修复；官方 `cartpole-balance` 5k steps；1.20M 参数，初始评估 239.9、最终 657.6，增长 174.1% | **上游最小复现完成**；不是 104-task 论文复现 | 其 dynamics/reward/value/policy prior 联合训练解释了为什么它能把模型误差转成控制；继续作为控制对齐强基线 |
| TD-MPC2 → 原始 5-DoF | 33-D adapter、硬 D3 测试、随机/方向性覆盖、1.2k/5k guided runs | **迁移链跑通，性能 No-Go** | 已定位随机数据零接触和 wrapper 覆盖策略的 bug；先用 strong teacher/DAgger 确认任务，再比较世界模型 |
| LeWorldModel/LeFlow 参考链 | 官方 PushT 环境、72.29 MB checkpoint、18,034,478 参数严格加载、弱专家采集、CEM rollout、10-episode paired screen；官方训练数据压缩包约 13.14 GB，未在限时内下载 | **checkpoint-planner-control 链复现完成；论文统计未复现** | offset25 仅方向性 4/10 对 1/10，`p=0.25`；说明需要足够数据与控制目标，不能用一次成功宣传复现 |
| DINO-WM | 完整代码审计；确认 953 MB checkpoint 和约 2.79 GB PushT 数据需求 | **资产/依赖审计，正式 evaluation 待完成** | 可作为视觉 latent world-model 参考；在原始臂 state-based 核心未过前不启动大下载/视觉迁移 |
| DyWA（ICCV 2025） | 完整仓库、预训练链接、训练协议和损失拆解；确认 323 个训练物体、PPO teacher 200k iterations、DAgger student 500k iterations、history + FiLM + impedance | **代码级复现审计；运行被 Windows/IsaacGym/CUDA 11.3 栈阻塞** | 其大提升不是一个 world-model head 产生；fair transfer 必须拆 teacher/DAgger、history、future loss 和 action space |
| fault-locomotion-isaaclab | 固定 commit、源码 `compileall`；审计 4,096 并行环境、五帧历史、29 种故障和 PPO/MoE 协议 | **代码编译通过；Isaac Lab 缺失，性能未复现** | 故障类型是电机失效而非硬位置锁定，只借鉴训练/消融规模，不作定量 baseline |
| DreamFLEX（ICRA 2025） | 项目页、视频、任务和消融协议审计 | **无公开完整代码，无法忠实复现** | 借鉴 fault-vector estimator、跨故障训练和真机报告结构，不引用本地性能 |
| IROS 2024 multi-joint-failure NPM | 论文、failure-constrained workspace、kinodynamic map、sim-in-loop planner 和真实试验协议审计 | **论文级复现设计；未找到源码** | 它直接占据“故障可行空间 + NPM 规划”，因此本方法差异必须落在三试验 paired effect/margin adaptation |
| Adaptive Compensation for Robotic Joint Failures | 检查公开仓库，只有少量环境文件、图片和视频，缺 Isaac/PPO/config/data/checkpoint | **发布物不足，报告数字不可独立复现** | 不把其 93.6% 当成本地 baseline；只列相关工作与复现边界 |
| PIN-WM / ActivePusher / CA-OED / ReDRAW | 论文、任务、数据、few-shot/active/residual 机制和可用资产审计 | **方法与差异化审计，尚未本地训练** | 用于确定 few-shot physics、主动探测、残差适配均不是单独创新；仅选择最贴近且可运行的 matched baseline |

复现的机器可读产物统一保存在 `results/reproductions/`，外部源码和补丁位于
`third_party/reproductions/` 与 `third_party/patches/`；总审计报告为
`reports/reproduction-first-audit-20260830.md`。没有达到论文统计规模的项目统一
写作 smoke、pipeline reproduction 或 code audit，不写作“成功复现论文性能”。

完整报告入口包括：

- `reports/paper-claim-evidence-audit-20260829.md`；
- `reports/reproduction-first-audit-20260830.md`；
- `reports/genkiarm-three-seed-interim-20260830.md`；
- `reports/ipwm-formal-invariants-and-guard-bound-20260829.md`；
- `reports/sfet-core-mechanism-sprint-20260830.md`；
- `reports/sfet-task-level-nogo-20260830.md`；
- `reports/icra-16-day-primary-platform-reset-20260830.md`；
- `reports/to-senior-2100-progress-20260830.md`。

### 4. 对学长提出问题的逐项回答

#### 4.1 “仿真太少”

这个判断对“论文有效证据”是成立的，但不是因为程序运行次数少。当前已有上万条
counterfactual branches 和大量诊断，问题是它们来自相关 prefix/action forks，且
集中在单一 Push、短 horizon 和开发 Gate；不能等同于大规模独立 episodes。

需要补的不是更多重复 smoke，而是：

1. 原始 5-DoF 主平台上的独立训练种子和完整闭环 Push；
2. D2/D3、未见锁定角度、摩擦、质量、目标和接触状态的正交覆盖；
3. 同数据、同 constrained IK、同候选、同规划预算的强基线；
4. action regret、终点误差和成功率，而不只报告平均状态 RMSE；
5. 第二机械臂和第二任务作为外部有效性，而不是替代主平台。

#### 4.2 “真机本身比较拉胯，只能堆仿真吗”

低成本 5-DoF 真机不会天然降低论文价值，但它只能承担与能力匹配的现实性验证，
不能承担高性能机械臂的控制上限。真机主实验冻结为固定夹爪低速 Push：intact、
D2、D3，预定义起点/目标、重复 trials、双 eye-to-hand 记录方块 XY、末端、关节
状态、接触和锁定漂移，全部失败保留。

`sim/assets/arm_push.xml` 只复制原始臂运动学链、关节范围和工具偏置；碰撞、
惯量、摩擦、执行器和接触参数尚未完成系统辨识，因此只能称“原始臂运动学任务
模型”，不能称精确数字孪生。真机数据优先用于三条故障校准轨迹和现实性检查；
性能主证据主要由严格仿真矩阵承担，但仿真不能冒充 sim-to-real。

#### 4.3 “换其他机械臂，能否说明泛化能力”

可以，但必须区分四层：

| 层级 | 合法结论 | 当前状态 |
|---|---|---|
| L0 接口兼容 | 同一代码处理不同 DoF/节点数 | 已通过 |
| L1 结构共享 | 同参数化机制和超参数可在两臂运行 | 局部成立；两臂共同训练 |
| L2 故障适配迁移 | 目标臂的故障数据不参加开发；允许其健康数据训练冻结的 pre-fault IPWM，测试时仅给结构、已知锁定和相同 3-shot，仍改善 object effect、ranking、regret | 未成立；object/contact 当前 0/3 |
| L3 闭环任务迁移 | L2 改善进入第二臂 Push/Grasp 成功率或终点误差 | 未验证 |

因此原始 5-DoF 必须是 primary platform；Panda 是首选 secondary external-
validity arm，GenkiArm 保留为第三平台/失败边界。只有冻结方法先在原始臂成立，
再以 target-arm fault-held-out adaptation、无 robot-specific adapter/head、无逐臂
调参、相同 3-shot 预算迁移，才能声称跨机械臂故障适配。目标臂允许使用自己的
健康数据训练冻结 IPWM；迁移的是 FCCM 适配规则与超参数，不是声称一套 IPWM
权重零样本跨 DoF。把两种臂的故障数据一起训练后只留一个关节，叫跨关节泛化或
multi-embodiment fitting，不叫 unseen-arm fault adaptation。

#### 4.4 “光 Push 不够，机械臂最好增加 Grasp”

同意增加，但不能把两个任务做成两个独立模型。统一问题定义为：

> 已知单关节锁定后，constrained IK 先生成故障可行接触候选；冻结 IPWM 与
> 三条故障轨迹只负责预测和重排这些候选的任务效果。

- Push 是主任务：候选为接触位置、方向、速度和短轨迹，效果为物块 XY/姿态、
  action regret 和 terminal goal loss；承担完整主表、消融和统计。
- Grasp 是压力测试：constrained IK 生成 pre-grasp、closure、lift 候选；同一个
  backbone、投影、3-shot adapter、ranking loss 和 fallback 预测接触保持、滑落
  和抬升结果。只允许 task cost/readout 不同。
- Panda scripted Grasp 5/5 只清除了任务可行性风险。只有故障下与强基线公平比较，
  才能作为方法证据。若 Grasp 需要另一套网络、另一种适配损失或 Contact-GraspNet
  才能工作，应降为附录，不把组件堆叠计作 IPWM 贡献。

#### 4.5 “idea 以前有人做过，如何差异化”

近邻已经覆盖以下单点，均不得单独声称创新：

- failure-constrained workspace、kinodynamic map 与 joint-failure NPM 规划：
  IROS 2024 `Exploring How Non-Prehensile Manipulation Expands Capability in
  Robots Experiencing Multi-Joint Failure`；
- few-shot Push 物理参数辨识与 MPC：ICRA 2025
  `Incremental Few-Shot Adaptation for Non-Prehensile Object Manipulation`；
- history-conditioned dynamics adaptation：IROS 2020 recurrent pushing 与
  ICCV 2025 DyWA；
- physics-informed few-shot world model：RSS 2025 PIN-WM；
- residual physics、主动采样和 uncertainty-aware planning：ICRA 2026
  ActivePusher；
- contact-aware optimal experiment design：RSS 2025 CA-OED；
- frozen backbone + latent residual adaptation：L4DC 2026 ReDRAW。

因此不能把“解析约束＋残差网络＋few-shot＋主动 probe＋排序损失”写成组件清单。
下一候选核心必须整体定义为：

> **对训练未见但已诊断的离散关节锁定，先在解析故障可行流形上生成接触候选，
> 再以冻结的故障前 IPWM 作为 paired counterfactual control variate；仅用三条
> 故障轨迹估计任务效果差分/候选 margin 的改变，直接降低 action regret；解析
> 投影、状态隔离和置信 margin 回退阻止学习分支破坏 carrier。**

暂称 **Fault-Conditional Counterfactual Margin IPWM（FCCM-IPWM）**。它学习的
不是完整故障转移、任意状态残差或锁定后的 IK，而是“故障如何改变可行候选之间
的控制边际”。可检验形式为：

`z0(a)=psi(F0^H(s,Pi_d(a)))`，
`r_d(a)=z_d(a)-z0(a)`，并直接拟合
`m_d(a_i,a_j;g)=m_0(a_i,a_j;g)+w_d^T(phi_i-phi_j)`。

若同样三条数据下 raw-command ridge、physics-ID、generic residual 或 history/FiLM
达到相同结果，则 IPWM prior 没有独立价值，FCCM-IPWM 必须 No-Go。论文不使用
“首次提出”或“5分创新”；只声称上述完整的故障干预—可行流形—paired task-
effect ranking 链条与已审计近邻不同，并通过 matched ablation 证明差异。

### 5. 两个扩展方向：并行预备，主 Gate 通过后才正式执行

核心顺序仍是：FCCM-IPWM 必须先在原始 5-DoF Push 的小 Gate 上成立。与此同时
并行完成两条扩展方向的资产和冻结协议；主 Gate 通过后立即启动两条正式矩阵，
主 Gate 失败则保留资产/可行性结果，不用广度包装无效机制。

#### 方向 A：跨机械臂 Push 泛化

- primary：原始 5-DoF `arm_push.xml`；secondary：官方 Panda；GenkiArm 为补充
  边界，不替代 primary；
- source-arm 用于开发 FCCM 适配规则；target-arm 的故障条件完全留出，但允许
  使用 target-arm 健康数据训练其冻结的 pre-fault IPWM；测试只新增运动学描述、
  已诊断锁定和相同三条故障校准轨迹；
- 两臂使用相同 adapter 结构/初始化规则、ranking、planner、候选数、超参数和
  计算预算；不要求不同 DoF 共用同一 IPWM 权重，禁止 robot-ID、故障专属 head
  和 target-arm 逐臂调参；
- primary metrics：object task-effect error、within-state Spearman、top-1 regret、
  terminal error、success、contact survival、locked-coordinate violation 和 latency；
- Go：原始臂主 Gate 先过；随后 target arm 的 D2/D3 至少 3/3 seeds 同向，
  regret 至少下降 50% 或 terminal error 至少下降 30%/成功率提高 20pp，并显著
  优于同数据 direct ridge；否则只能报告 L0/L1，不能写跨臂泛化。

#### 方向 B：Push + Grasp 跨任务验证

- Push 保持唯一主任务；Grasp 使用 Panda 已通过可行性的 top-down grasp + short
  lift，原始臂只在夹爪几何和执行器确认后加入；
- 两任务共享硬投影、constrained IK、冻结 IPWM、三条校准轨迹、margin adapter、
  fallback 和超参数，只替换 task readout/cost；
- Grasp 主要指标为稳定双指接触、短抬升成功率、drop/slip regret、末端/物体误差
  和锁定漂移；必须在 constrained IK 接触率相当的前提下比较；
- Go：D2/D3 至少三个独立 seeds，方法相对最强同数据基线成功率提高 20pp 或
  drop/error/regret 降低 50%，且去掉 IPWM prior 后收益显著消失；否则 Grasp 仅作
  feasibility/附录，不把脚本 5/5 写作方法结果。

### 6. 分阶段仿真与真机矩阵

#### Stage 0：核心小 Gate，立即执行

- 原始 5-DoF、D2/D3、完整接触 Push 轨迹；
- 三个开发 seeds，三条故障校准轨迹，held-out targets/angles/physics；
- 所有方法共享 constrained IK；
- 方法：nominal IPWM、direct 3-shot ridge/physics-ID、generic full residual、
  history/FiLM、SFET 历史边界、FCCM-IPWM；
- 主指标：top-1 regret 和 terminal error；次指标：task-effect RMSE、Spearman、
  contact survival、锁定违例、latency；
- **Stage 0A 开发筛选门（8/31 14:00）：**D2/D3 的 3/3 开发 seeds 方向一致，
  相对最强 3-shot baseline 的 regret 至少下降 30%，terminal error 同向改善，
  lock violation 为零，且去 IPWM prior 后收益下降；只决定是否继续完成实现；
- **Stage 0B 最终确证门（9/5 12:00）：**D2/D3 均 3/3 seeds 相对最强同数据
  baseline 的 regret 下降至少 50%，且闭环终点误差下降至少 30%或成功率提高
  20pp；去 IPWM prior 后收益显著消失；只有该门通过才升级为论文核心；
- No-Go：只降 RMSE、不降 regret；只 D3/单 seed；或 constrained IK 已把严格任务
  做到近饱和。失败后禁止再收缩到特定 seed/目标，也不启动昂贵确认矩阵。

#### Stage 1：Stage 0A 后做原始臂确证，Stage 0B 后才做外部扩展

原始臂 Push 矩阵用于完成 Stage 0B，不以 Stage 0B 已通过为前提；Panda Push
与 Panda Grasp 只有在 Stage 0B 通过后才进入正式统计。

最小确认量按每个方法计：

- 原始臂 Push：`intact/D2/D3 × 3 seeds × 5 unseen targets × 5 repeats`；
- Panda Push：相同冻结矩阵，作为 target-arm fault-held-out 外部有效性；
- Panda Grasp：`intact/D2/D3 × 3 seeds × 5 unseen cube poses × 5 repeats`；
- 至少四个公平方法时，总计不少于 `2,700` 个 evaluation episodes。每个 repeat
  必须使用预注册且不同的 reset、初始姿态、physics/noise ID；同一训练 seed、
  target 或 physics 组内仍视为相关样本，不把 2,700 当作 2,700 个完全 i.i.d.
  样本；置信区间按训练 seed 与 target/physics 组做分层 cluster bootstrap；
- 摩擦、质量、执行器强度和锁定角度采用预注册正交/held-out 组合，不能把多个
  相关 branch rows 冒充独立 episodes。

#### Stage 2：原始 5-DoF 真机现实性验证

- 两台相机均为固定 eye-to-hand；第二台为水平外置相机；
- intact、D2、D3，固定夹爪、一个刚性方块、低速可重复 Push；
- 先采三条安全校准 Push，再在未见目标上比较 constrained IK carrier 与冻结方法；
- 报告每次方块终点、是否接触、锁定漂移、执行时间、异常/中止和原始视频；
- 真机 Grasp 先做安全与可行性，不在没有公平对照时承担核心主张；
- 真机结果证明现实性，不替代仿真消融和统计，也不因硬件性能有限删除失败。

### 7. FCCM-IPWM 的最小理论和消融合同

要避免“又是组件组合”，至少同时完成以下三项可验证性质：

1. **控制变量样本效率。** 测量名义/故障任务效果相关性，验证 paired residual 的
   方差随相关性下降；不得只引用标准 control-variate 公式而不测量条件。
2. **预测到控制的 margin bound。** 若候选 pair 的 margin 预测误差上界为
   `epsilon`，则只有预测优势超过 `2*epsilon` 时接受修正，否则回退 carrier；
   同时报告 coverage、acceptance rate 和 accepted-set regret，防止“永远回退”
   获得空洞安全保证。
3. **IPWM 必要性。** 同三条数据比较：去 IPWM prior、打乱 intact/fault pairing、
   raw action 代替名义 task-effect 坐标、full-state RMSE 代替 margin objective、
   去状态隔离和去 fallback。任何同容量简单方法得到相同收益均判核心归因失败。

这里的 `paired` 冻结为同一个实测初始状态、候选轨迹 ID 和目标：名义效果
`z0` 由冻结 pre-fault IPWM 从该实测状态滚动同一解析投影候选得到，故障效果
`zd` 由 fault trial 实测得到。训练不要求沿途存在逐帧完全相同的真实 intact
轨迹；额外采集的同 reset/intact 重复只用于校验 pre-fault IPWM 和估计控制变量
相关性，不能偷换成第四条故障校准数据。

### 8. 给学长的冻结回答

> 这几天并非只跑了少量仿真，而是完成了五种子预测审计、双臂 2,880 条动作
> 反事实分支、14,400 条主动探测记录、90 个原始臂载体 episodes、三种子 Genki
> 确认、跨臂/接触/混合模式 Gate、Panda Grasp 与双 eye-to-hand 可行性、TD-MPC2
> 和 LeWM 本地复现以及 SFET 实现。大量 No-Go 说明此前问题定义把运动学可达性、
> 接触动力学和动作排序混在了一起，而不是没有做工作。
>
> 接下来不再堆组件。原始 5-DoF 仍是主平台，所有方法共享 fault-aware
> constrained IK；新核心只研究三条故障轨迹能否利用冻结 IPWM 修正可行候选
> 的任务 margin。仿真广度用两条并行外部证据补齐：一条是冻结方法迁移到 Panda
> 的 Push，另一条是同一机制迁移到 Grasp。两条都做，但只有原始臂核心 Gate 通过
> 后才扩为正式统计矩阵。
>
> 我们不把跨臂接口、脚本 Grasp、相机可见性或失败实验包装成方法成功。与已有
> joint-failure planning、few-shot physics adaptation、history-conditioned world
> model 和 active probing 的差异，必须由“故障可行流形＋预故障 IPWM paired
> control variate＋三试验 task-margin correction＋解析状态隔离/回退”整条机制和
> matched ablation 共同证明；如果 simple ridge 或 physics-ID 同样有效，就主动
> No-Go，不声称首次提出。

### 9. 禁止性表述

- 禁止继续称 constrained IK 为 oracle；
- 禁止把 SFET 合成 3-shot `41.4%` 写成机器人恢复成功；
- 禁止把 ordinary ensemble 的 `30.74%` 归因于 IPWM 结构；
- 禁止声称跨机械臂、Grasp、视觉或真机已经验证；
- 禁止把 `arm_push.xml` 称为精确动力学数字孪生；
- 禁止把 No-Go 说成“提高真实性所以增加录用率”；
- 禁止把 carrier-relative 状态隔离写成相对真值安全或任务不退化；
- 禁止把相关 counterfactual rows 写成独立大规模 episodes；
- 禁止隐藏 5-shot、失败 seed、失败任务或改变看过结果后的门槛；
- 禁止使用“首次提出”“5分创新”“显著优于 SOTA”，除非有直接查新、matched
  baseline、独立统计和闭环证据。

### 10. 当前停止条件和论文主张

在 FCCM-IPWM 通过 Stage 0 之前，论文正文只允许声称：

> 对已诊断单关节锁定，SI-IPWM 用解析投影保证硬约束可行性，并通过私有分支和
> 选择性发布阻止物体动力学适配改写 carrier 的机器人状态；旧简化 Push 仿真
> 显示该隔离可在部分物理 OOD 单元保留 object-prediction 收益，但实际 GenkiArm
> 三种子尚未证明性能优越性。

若 Stage 0 失败，停止把新 margin 机制包装为核心；跨臂和 Grasp 只保留为工程/
任务可行性资产，论文评分不超过当前边界。若 Stage 0 通过，才依次形成：原始臂
Push 主表与消融、Panda target-arm fault-held-out Push、Panda/原始臂 Grasp 压力测试、
原始 5-DoF 真机现实性证据。任何阶段都保留全部失败。

### 11. 以 9 月 1 日真机为锚点的 16 天冲刺计划

本日程按 2026-08-30 晚冻结、2026-09-15 留作提交缓冲安排；若投稿系统的精确
截止时间相差一天，只平移提交缓冲，不改变各科学 Gate 的先后关系。资源比例冻结为：

- 原始 5-DoF Push 核心、真机和消融：`55%`；
- Panda 跨臂 Push：`10%`；
- Panda/原始臂 Grasp 外部有效性：`10%`；
- 统计、复现、图表、论文和提交包：`25%`。

Push 承担核心创新；跨臂和 Grasp 各只回答一个外部有效性问题，不能新增任务专属
核心网络，也不能用广度掩盖原始臂核心 No-Go。

#### 11.1 9 月 1 日前必须解除的两个真机硬阻塞

1. 当前部署脚本和配置仍命名为 `eye_in_hand`，而真实设置是两台固定
   **eye-to-hand** 相机；`config/deployment/eye_in_hand_aruco_v1.yaml` 中
   `marker_size_m`、`camera_matrix`、`distortion_coefficients` 仍为空，并且只有
   单相机索引。8 月 31 日必须建立左右两台 eye-to-hand 的独立内参/外参配置、
   公共桌面参考坐标、时间同步和原始视频保存；旧 eye-in-hand 名称与几何不能
   直接用于正式数据。
2. `hardware/safety_limits.yaml` 的毫安换算上限仍为空，只存在已测 raw-current
   中止值。自动运动前必须运行 readiness audit，确认串口、ID 映射、急停、
   feedback timeout、温度、raw current、锁定漂移和机械支撑全部为绿；不能因
   赶进度绕过空安全字段。正式现场沿用 `50 C`、current raw `400`、锁定漂移
   `3.5 deg`、反馈超时 `250 ms` 和单次锁定保持不超过 `10 s` 的保守中止边界。

#### 11.2 每日关键路径、产物与硬 Gate

| 日期 | 主任务 | 当日必须交付 | 硬 Gate 与失败后动作 |
|---|---|---|---|
| **8/30（D-16）** | 停止改题，冻结科学合同 | FCCM-IPWM 问题、公式、D2/D3、3 条校准轨迹、候选集、5 个未见目标、主指标、基线、消融和成功阈值；提交本次 Plan/报告 | 23:30 后禁止再引入新 head/flow/graph。完整轨迹候选若 nominal top-1 regret 仍小于 `2 mm` 或小于最佳动作 margin 的 20%，停止“巨大闭环优势”措辞，不靠筛目标制造空间 |
| **8/31 上午（D-15）** | FCCM 最小实现和原始臂 Stage 0A 开发筛选 | `IPWM nominal effect + paired 3-trial margin correction + confidence fallback`；D2/D3 × 3 dev seeds；constrained IK、direct ridge/physics-ID、generic residual、原 IPWM 对照 | 14:00 Stage 0A：D2/D3 方向一致、相对最强 3-shot baseline regret 至少下降 30%、terminal error 同向、lock violation=0、去 IPWM prior 后收益下降。失败不现场包装方法，18:00 停止结构调参 |
| **8/31 下午** | 真机全链 dry-run | 双 eye-to-hand 内外参、桌面/方块 marker、共同同步事件；J1--J5/命令/温度/current raw/视频/物块 XY/contact/abort logger；trial manifest、随机顺序、复位模板、空结果表、双盘备份脚本 | 18:00 `HW0`：日志字段、两相机、急停、动作语义、constrained IK 接触、方块复位任一不通过，9/1 降级为系统辨识/视觉数据采集，不执行方法闭环 |
| **9/1（D-14，真机日）** | 优先带回不可由仿真补造的真实原始证据 | 按下节 T0→T1→T2 顺序采集；每条 trial 独立视频、CSV/JSON、manifest、hash；最后 45 分钟只验文件和双份备份 | 中午 `HW1`：同步有效、日志完整、锁定安全、constrained IK 各条件接触率至少 4/5。否则取消方法比较，保留标定和失败边界。当天无完整重复则只能标 hardware pilot |
| **9/2（D-13）** | 真机数据审计、冻结与仿真校准 | 所有 trial 先按预注册规则判 valid/invalid，再解盲方法标签；输出 contact、terminal error、ranking/regret proxy、漂移、温度、abort；只用 calibration split 拟合延迟/执行器尺度/接触参数 | 20:00 `HW2`：标定模型只有 held-out trajectory error 至少降低 30% 才称 calibrated task model；否则仍称运动学任务模型。真机无同向方法收益则降为 feasibility/boundary，不改阈值重算 |
| **9/3--9/4（D-12--D-11）** | 原始 5-DoF 核心 Push 主矩阵 | D2/D3 × 3 dev seeds × 5 unseen targets × 5 repeats；每次 repeat 使用不同 reset/physics/noise ID；完整轨迹候选；一个 held-out friction/latency 组合；0/1/3/5-shot 曲线；全部方法共享 IK/candidate/compute | 9/4 22:00 初裁决；只 RMSE 好、不降 regret/闭环，或只 D3/单 seed 通过，均为核心 No-Go |
| **9/5（D-10）** | Stage 0B 最终确证和结构冻结 | 主表草案、paired rows、关键消融；相对最强同数据 baseline：regret 至少 -50%，terminal error 至少 -30% 或 success +20pp，D2/D3 和 3/3 seeds 同向，lock violation=0 | **12:00 是新机制最后停止线。** 失败后不再换机制、seed、目标或任务；回到 SI-IPWM 状态隔离/预测窄稿或决定延期。通过后才扩正式跨臂和 Grasp |
| **9/5--9/7（D-10--D-8）** | 方向 A：Panda 跨臂 Push | source/target arm 协议冻结；同 adapter、loss、planner、3-shot 和超参数；Panda 两个锁定、3 seeds、5 targets；Genki 只作补充边界 | 9/7 12:00 停止。只有接口运行、robot RMSE、2/3 seeds 或 object/contact No-Go 都不算泛化；失败用一张边界表结束，不训练共享大模型救场 |
| **9/7--9/9（D-8--D-6）** | 方向 B：Push→Grasp 跨任务 | Panda 方块 top-down pregrasp/closure/short-lift；同硬投影、IK、IPWM、3-shot margin adapter、fallback，只换 task readout；D2/D3 × 3 seeds | 9/7 中午先验证 constrained IK 接触率≥80%且候选结果非平凡；全 5/5 或全失败都没有世界模型比较空间。9/9 12:00 停止 Grasp 扩展，禁止增加专用 backbone |
| **9/9（D-6）** | 统一主张与图表口径 | 只允许三种论文状态：核心 Go+至少一个扩展 Go；仅核心 Go；核心 No-Go。冻结 primary metric、success threshold、seed、target split 和机制版本 | 23:00 后不得根据图表需要改实验定义；跨臂/Grasp 不能替核心补分 |
| **9/9--9/10（D-6--D-5）** | untouched confirmation 与全部数字冻结 | 对冻结方法跑 2 个完全未触碰种子，使原始臂核心最多形成 5-seed 证据；完成 no-IPWM、shuffled pairing、full-state-vs-margin、no-fallback、probe 消融；bootstrap CI、effect size、latency | 9/10 24:00 后禁止新增实验分支。确认方向不一致或 CI 覆盖大幅负效应则主动降级主张 |
| **9/11（D-4）** | 统计和证据账本 | 唯一 claim→artifact→metric→status 表；核验每个 shot 对应完整 trajectory 和 transition 数；主表、shot 曲线、失败图、方法图、约束图定稿 | 原始 rows、失败和 5-shot 饱和边界缺失则不进入写作冻结 |
| **9/12（D-3）** | 全文重写和差异化 | 题目/摘要围绕一个机制；Push 为完整证据，跨臂和 Grasp 各一 panel；正面对比 IROS24 failure-NPM、ICRA25 few-shot pushing、PIN-WM、ActivePusher、DyWA、ReDRAW | 当天冻结 scope、related work、limitations；不得使用 first/SOTA/5分创新语言 |
| **9/13（D-2）** | 内部 ICRA/CCFA 盲审 | 完整 PDF 对 novelty attribution、baseline fairness、统计、双 eye-to-hand 描述、真机边界、图表可读性逐项评分 | 只允许解释、排版和已冻结实现 bug；审稿意见不能触发重选数据 |
| **9/14（D-1）** | artifact 与提交包冻结 | 一键复现说明、config/hash、raw rows、失败 ledger、匿名检查、最终 PDF/视频/附件 | 18:00 code/data/results freeze；之后只修格式和文字 |
| **9/15（D0）** | 提交缓冲 | 作者/匿名/页数/参考文献/附件/上传校验，至少提前 6 小时上传首版 | 不再重跑实验，不让上传故障吞掉全部缓冲 |

#### 11.3 9 月 1 日真机数据包：T0 必须完成，T1 尽量完成，T2 有条件完成

**T0：无论新方法是否通过都必须带回。**

1. 真机身份与几何：J1--J5 ID、正方向、零位、软/硬限位、连杆、固定夹爪
   实际接触点、底座到桌面坐标；方块尺寸/质量、桌面材质、全景与近景照片。
2. 动作语义和执行器响应：明确归一化 action 到 ST3215 位置命令、周期和保持
   时间；intact/D2/D3 做低幅无接触阶跃或安全 PRBS，记录 command、q、qdot、
   current raw、温度、延迟、死区和回差。仿真 action 不能未经 adapter 验证直接发送。
3. 锁定真实性：D2/D3 各至少一个主锁定角和一个安全边界角，记录静止保持与
   邻接关节低幅动作时的漂移、温升、电流和保护事件；使用可重复 servo hold，
   不做破坏性机械卡死。
4. 双 eye-to-hand 原始视觉：左右相机分辨率/FPS/曝光、内外参、9 点静态桌面
   网格、夹爪遮挡和接触前后视频；使用共同 LED/声音事件或单调时钟对齐两路
   视频与舵机日志。即使融合失败，也保留全部原始帧供离线标注。
5. 真实接触数据：intact/D2/D3；每种先做冻结的校准 probes，再做独立未见方向/
   幅值的 evaluation pushes；保存初末图、方块 XY、q/qdot、commands、contact、
   abort 和完整 manifest。不能只拍成功视频。

建议的最低可复用 Push 包为：

- intact：10 条 baseline Push；
- D2：3 条 calibration Push + 至少 10 条 evaluation Push；
- D3：3 条 calibration Push + 至少 10 条 evaluation Push；
- calibration 与 evaluation 使用物理分开的目标/动作；试验顺序随机交错，人工
  复位使用定位模板，并以视觉实测初态为准。

**T1：强基线，T0 和安全门通过后。**

- fault-aware constrained IK 在 intact/D2/D3 的相同两个目标上至少 5 次重复；
- 与 fault-unaware/hard-mask 只作为失败参照，论文主要比较必须是强 constrained IK；
- 如果 constrained IK 已接近 100%，真机主指标改用预先冻结的 terminal error/
  action regret，而不是事后收紧 success 半径制造差异。

**T2：冻结方法对照，只在 8 月 31 日完整 dry-run 通过后。**

- 只比较 constrained IK carrier 与一个冻结 FCCM/SI-IPWM checkpoint；相同起点、
  目标、候选、计算预算和视觉来源，paired/interleaved；
- 不在现场训练、换目标、改锁定角、调成功阈值或根据首批结果选择 seed；
- 若在线方法未准备好，使用 T0 候选库在 9 月 2 日做严格离线排名评估，不能把
  离线重放写成真机闭环。

Grasp 在真机日只做 T0 之后的可选最小可行性包：固定方块 top-down contact/close/
short-lift，intact/D2/D3 各最多 5 次。原始臂无独立 J6 姿态自由度，夹爪几何和
接触尚未正式标定；若任一安全/姿态 Gate 失败，立即停止，把方法级 Grasp 留给
Panda 仿真，不牺牲 Push 的重复数和数据完整性。

#### 11.4 真机现场时间比例停止规则

- 用掉现场时间 **25%** 时，若相机同步、完整 logger、急停、单关节 mapping、
  无接触动作语义仍未全绿，正式任务降级为 T0 系统辨识/视觉采集；
- 用掉 **50%** 时，若 constrained IK 在 intact/D2/D3 不能各达到至少 4/5 接触，
  或出现锁定漂移、过流、过热，取消方法比较；问题首先归为可达性/安全，不能
  让世界模型背锅；
- 用掉 **70%** 时，若 paired 日志或视频完整性仍不稳定，停止新增方法 trial，
  只补齐最低条件；
- 最后 **45 分钟** 禁止新实验，只逐条打开视频/CSV、生成 SHA256、完成 manifest、
  双盘复制和现场照片；
- 任何协议中途修改均新开 session，不把修改前后 trials 混进同一统计表。

#### 11.5 真机证据等级与降级路线

| 等级 | 9 月 1 日实际完成 | 论文合法用途 |
|---|---|---|
| A | T0/T1 完整，冻结方法也 paired 运行 | 小样本真机现实性表；主结论仍由仿真承担 |
| B | T0/T1 完整，方法链失败或未准备 | 强基线、模型校准和 failure boundary；不声称真机方法优势 |
| C | 仅 T0 完整 | 硬件/视觉/执行器数据集与 sim-to-real gap 审计；论文保持 simulation-only |
| D | 安全、通信或同步未通过 | 仅几何、静态视觉和遥测；不做接触，不用成功视频填主表 |

9 月 2 日中午必须完成 valid/invalid ledger；9 月 3 日晚必须确定真机证据等级和
论文范围，不能拖到写稿时再解释。

#### 11.6 16 天内明确不承诺的事项

- 同时从零发明新机制、完成完整理论、五种子主表并获得显著真机优势；
- 建成完整动力学数字孪生，包括惯量、摩擦、回差、执行器和接触全辨识；
- 两台 eye-to-hand 端到端视觉世界模型与鲁棒闭环；
- 原始臂、Genki、Panda 三臂严格 target-arm fault-held-out 全矩阵；
- Push 与故障 Grasp 都完成同等规模的主表、消融和真机统计；
- 忠实复现多篇近期顶会的全部数据规模，或训练千万级多任务世界模型；
- 用 5 次真机 trial 宣称统计显著，或承诺 ICRA 4+/5、5分创新和录用。

最晚停止日期冻结为：**9/5 12:00 停止核心机制变化；9/7 12:00 停止跨臂扩展；
9/9 12:00 停止 Grasp 扩展；9/10 24:00 冻结全部数字；9/12 冻结论文主张；
9/14 18:00 冻结最终提交包。**

### 11.7 8 月 31 日严格归因更新：大指标与边界（权威覆盖项）

统一使用三个预定开发 seed（7/17/27）、每 seed 400 组、每组 128 个候选、完整
checkpoint 严格加载。以下数字不得与旧的不完整 checkpoint 评测混用。

- **稳定控制相关大指标存在，但不属于选择性结构独占。** 同容量全局残差相对
  nominal WM 的 top-1 regret 平均降低 **19.76%**（范围 10.66%--27.86%，
  3/3 seeds 同向），候选终点误差平均降低 **4.04%**（3/3），成功率平均
  +1.58 pp（2/3 正、1/3 持平）。与此同时接触候选响应 RMSE 平均恶化
  **270.04%**（0/3 改善）。这支持“平均预测误差与动作选择/任务结果解耦”的
  诊断结论，而不能写成全面预测提升。
- **选择性 IPWM 相对同容量全局残差归因 No-Go。** IPWM 的 regret 相对全局
  残差平均为 **-2.07%**、endpoint 为 **-0.32%**、Spearman 平均仅
  +0.00037；虽然 regret 和 endpoint 各有 2/3 seed 微弱改善，但幅度小、
  Spearman 仅 1/3 改善，且 full-state 与 selective publication 在三个 seed
  完全相同。正文不得声称路径选择性产生现有 18%--20% 优势。
- **解析硬投影得到干净的结构性 Go。** 移除投影后，锁定关节最大位置违例为
  0.077--0.153 rad（平均 **0.116 rad / 6.63°**），最大速度违例为
  0.384--0.660 rad/s（平均 **0.539 rad/s**），3/3 seeds 均非零；启用投影后
  两类违例在 3/3 seeds 均严格为 **0**。任务指标几乎不变，因此该贡献应准确
  表述为结构约束/安全保证，而不是成功率增益。

机器可读证据：

- `results/final/primary-global-matched-ablation-3seed.json`
- `results/final/primary-projection-ablation-3seed.json`
- `results/final/large-advantage-metric-audit.json`

截至本节，最诚实的主线是：**解析投影保证故障约束；控制相关适配稳定改善候选
动作 regret；但选择性影响建模尚未通过同容量归因，且预测 RMSE 与控制收益存在
系统性冲突。** 后续只能补确认性证据、真机和写作，不得换 seed、删掉全局残差
或把该 No-Go 隐藏到附录。

为避免“寻找大指标”退化成挑 seed、换阈值或混用简化臂结果，新增机器审计将指标
分成 PRIMARY_ATTRIBUTABLE、PRIMARY_CONTROL_RELEVANT、诊断性和禁止作为核心
创新四类。当前允许进入摘要的组合固定为：**锁定结构违例消除 100% + top-1
regret 平均降低 19.76%（3/3 seeds）**。历史选择性预测在 50% coverage 下
RMSE 降低 50.50% 只能进入诊断/附录，直到固定 rollout 深度确认；三成员集成
约 30.74%--33.55% 的收益属于通用 ensemble 基线，不能归因给 IPWM。

真机的大效应量口径也已在看结果前冻结：除绝对成功率差和配对 bootstrap CI 外，
分析器同时输出 endpoint 相对降低、失败率相对降低，以及“候选救回 baseline
失败/候选破坏 baseline 成功”的 discordant pair 计数。相对失败率必须与原始
失败率、绝对百分点和计数同表报告；baseline 零失败时输出 null，禁止写无穷提升。

正式真机试验新增 fail-closed preflight：固定 schedule 必须保持 SHA-256
`79139bca...f4510d`、25 对/50 次及 intact/D2/D3=5/10/10 对；会话清单、双
眼在手外相机标定/同步视频、三类日志目录、两份备份、安全检查和冻结签字必须
完整且路径真实存在。审计未输出 `FORMAL_TRIALS_MAY_START` 时不得开始正式试验。

8 月 31 日动作接口复核发现一个必须公开的真机边界：仿真模型的 action 是 0.005 s
MuJoCo motor 广义力，而原始臂接收 raw tick 目标位置；仓库尚无经过实验验证的
映射和冻结候选动作库。因此真机证据分为 Level A 固定低速轨迹的锁定/可达/接触/
推块机制验证，以及 Level B 学习方法比较。Level A 可在普通安全门禁后执行；
Level B 还必须提供 action bridge、低幅度验证日志、共同候选库哈希和逐 trial
模型选择 ID。未满足时不得把手工轨迹标成 nominal/global 方法结果。完整审计见
`reports/real-robot-action-interface-audit-20260831.md`。

Level A 已有独立随机表生成器和正式汇总口径：必须先人工验证并填写 intact/D2/D3
三个真实固定轨迹 ID，随后生成各 10 次、共 30 次的随机顺序；CSV 中 method 永远
为 `fixed_safe_trajectory`。统一分析器按 condition 输出锁定误差、reach/contact、
endpoint、success、abort 和 failure code，且只有每种条件至少 10 条有效记录并
通过双视频/控制日志文件检查才标记 formal。该表仅支持物理机制/可行性结论。

Level A 的论文资产也已闭环：严格 JSON 可直接生成双面板矢量 PDF（终点误差与
reach-contact-success）和 LaTeX 表（含锁定最大误差、abort、样本量），图底部与
表注固定声明 `Physical feasibility only - no learned-method comparison`；无有效
真机证据时生成器 fail closed，禁止用占位数字进入主稿。

为防止填写测量后无法证明“未删失败/未换轨迹”，现场必须分别保存冻结空白
schedule 与 completed trial log。新增逐行审计只允许填写测量、视频和日志字段，
并强制 trial_order、condition、position、method、trajectory_id 与冻结表完全
一致；缺行、增行、重复 order 或事后替换轨迹均 fail。该审计与原始文件有效性
门禁串联，两者均 PASS 后才能生成论文资产。

### 11.8 8 月 31 日独立 ICRA/CCFA 严格复评

当前七页稿、严格机器汇总、来源台账、聚焦契约测试和计算成本台账接受统一量表
复评。官方 ICRA 2027 规则已核验：完整论文最多八页（含参考文献）、双匿名、
截止 2026-09-15 23:59 PST。当前篇幅七页，格式长度通过，但匿名 class/PDF
metadata 仍需最终检查。

当前客观结论为：

- 六阶段图补入后的项目综合约 **3.6/5**，不是 4+/5；
- ICRA 决策尺度约 **5/10，weak reject / borderline**；
- 技术正确性与可复现性约 4/5；
- 新颖性、证据广度、表达聚焦度约 3/5，形成当前评分上限。

决策级 P0 缺口严格固定为：

1. 原始 5-DoF 真机 paired Push 原始证据、有效性 ledger、锁定/接触/终点误差；
2. 一个完全冻结且不再调参的 untouched confirmation 包；
3. 接受选择性结构归因 No-Go，将论文明确定位为六阶段诊断研究；
4. 六阶段证据图，替代只突出 state isolation 的视觉中心（已完成，Fig. 2）。

满足前两项且方向与论文一致，再完成图表与匿名检查，客观上才可能接近
**3.9--4.1/5 或 ICRA 6/10**。仅润色、堆附录或继续发明网络不能使当前稿达到
4+/5。完整报告：`paper/ccfa-review-reports/current-icra-review.md`。
