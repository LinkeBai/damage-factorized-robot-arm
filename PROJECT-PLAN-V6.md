# Project Plan V6 — Intervention-Projected World Model (IPWM)

**最后更新**：2026-08-24
**当前事实源**：本文件、`reports/icra-senior-review-remediation-20260824.md`、`runs/g2_r0_icra_audit_20260824/summary.json`
**仓库状态**：G2-R 已完成五种子、30轨迹、逐窗口不确定性复审；一个free-joint工程门失败已披露。G3真机任务证据仍未完成，尚未投稿。

## 1. 项目问题与最终主张

目标是在只知道锁定关节编号/锁定角、没有力传感器、只有电流/位置速度和手眼视觉的条件下，对未见锁定拓扑与物理组合进行安全的少样本世界模型适配，同时保持普通物体预测不明显退化。

最终论文简称为 **Intervention-Projected World Model (IPWM)**，物理上下文与滚动风险是其机制而非继续堆入缩写。它属于 BT-DPWM 研究轨迹，但不是严格的纯 forward block triangle：接触任务中机器人基座保留 contact-aware 条件；新增干预残差在 stop-gradient 边界后沿机器人/推子几何传播。

核心贡献：

1. **解析损伤投影**：锁定坐标由解析几何/状态投影硬约束，constraint violation 恒为零；网络只学习自由坐标动力学。
2. **支持感知干预残差**：冻结 contact-aware robot block 与 shared object base；显式传播 pusher 几何，并用低秩 latent residual 只修正训练支持集外的损伤干预。
3. **物理上下文与可逆安全门**：从 K 条 state/action 过渡估计 8D observable physical-context posterior；`h(z)-h(0)` 保证 K=0 精确回退；support gate、8-step grace、depth-risk ramp、hysteresis 和 conformal interval 限制更新。

## 2. 冻结模型与评估协议

- robot：Z69 contact-aware block；已知锁定拓扑只进入解析 projection，不向 shared 的未训练 topology 输入列写入随机真实值。
- object：shared compact base + rank-16 geometry residual + rank-32 bridge aligner + rank-8 physical-context intervention residual。
- context：Z65 observable posterior；K25 规则为 posterior scale 1.38、8 步 grace、depth ramp 0.06、centered zero bypass。
- 适配：shared+analytic-projection 与 BT 使用同协议 rank-8 adapter、相同 K25 支持数据、相同 rollout 预算；BT 采用 adapter-before-object 顺序。
- intact 时 `damaged = mask.sum(-1)>0.5)，干预路由强制关闭。
- seeds：7、17为开发，27为冻结确认，37/47为不调参审计扩展；domains：D3 composition、D3 mixed-unseen；horizons：H10/H25/H50。
- baseline：compute-/protocol-matched `shared + analytic projection + rank-8 adapter`。
- 门：locked violation = 0；object点估计工程门为2%；free 目标是不超过5%回退（已有一个约6%失败，不能宣称全过）；IID/seen object历史控制差1.72%，pusher历史绝对差1.081 mm。

## 3. G2-R 已完成证据

### 3.1 严格 matched-adapter 结果（权威）

相对 matched shared 的 object RMSE 改善（composition / mixed-unseen，H10/H25/H50）：

| seed | composition | mixed-unseen |
|---|---|---|
| 7 | +11.82 / +4.90 / +37.82% | +15.10 / +31.54 / +36.49% |
| 17 | +6.47 / +13.37 / +12.72% | +8.83 / +16.45 / +4.10% |
| 27 | +7.63 / +2.04 / +19.12% | +9.04 / +3.22 / +5.33% |
| 37 | +8.41 / +20.51 / +32.09% | +10.83 / +21.35 / +2.26% |
| 47 | +10.63 / +36.06 / +31.13% | +12.54 / +39.57 / +26.51% |

30/30点估计超过冻结2%工程门，最小增益 **2.0434%**。每格trajectory-cluster 95% CI下界均大于0，最小下界1.274%；六个跨seed均值CI也均大于0。所有locked violation为0，但seed7 mixed-unseen H50 free约退化6%，未通过5%门。

### 3.2 控制与安全证据

- 历史matched IID/seen object最大绝对差1.72%；不得再表述为全部free在5%内。
- pusher 最大绝对差 1.081 mm。
- K0 通过 centered gate 精确回退到 base map。
- conformal physical-context interval 的 dimensionwise MACE = 0.0289。
- full test suite：248 passed。

### 3.3 结构消融

seed27、每域30轨迹的matched消融显示：

- 去geometry：mixed-unseen为−7.44/−15.42/+6.75%。
- 去latent：为+4.40/−2.08/−50.86%。
- 去intervention：仅+0.01/+0.11/+0.43%。
- seed17无depth-risk的H50 mixed-unseen为−3.78%；完整规则为+4.10%。

这些结果支持不同组件对应不同失败区间，不能归因于adapter-only；不支持“完整模型在每个消融单元都最好”的过强表述。

### 3.4 历史结果的正确位置

早期 projected-only seed27 mixed-unseen H25 为 **1.9579%**，是历史近失，不能圆整为通过；最终 matched-adapter 协议已取代它作为公平门结论。旧 Z65/Z69、Z75、Z80/Z81 数字只作为开发轨迹或校准背景，不再作为当前版本号或独立主张。

## 4. 已否决路线（保留审计，不再继续切换）

原始 DFWM residual latent、history/FiLM、active latent optimization；dual-expert/DE-DWM 风险分歧；strict forward block triangle、tangent-normal hidden masking、DPP internal projection；无条件 scale/clip/decay 搜索；覆盖整个 object head 的 residual；asymmetric stitch、factorization-only、topology-conditioning-only 归因；raw posterior spread 作为 rollout-risk 排序器；“每增加 K 必然单调改善”与“全面显著胜 shared”的主张，均已判 No-Go 或缺乏独立因果证据。

这些失败用于确定接触反作用、解析约束、组件归因和风险校准的边界，并直接约束当前模型。

## 5. 阶段门

### G0 — 物理与接口基线：完成

MuJoCo/URDF、锁定协议、ST3215 电流/位置/速度、手眼视觉、安全急停和 dry-run 采集脚本已完成。

### G1 — 原始机制与 robust pivot：完成

DFWM 原始路线 No-Go；robust zero-shot ensemble 作为历史强基线完成，只用于动机与 matched baseline。

### G2-R — 仿真门：完成

五seed strict matched-adapter、K0/K25、逐窗口raw rows、聚类CI、结构消融、failure ledger与hash manifest已完成。结论是 **object-specific adaptation PASS with one disclosed free-joint gate miss**，不是所有安全/控制门通过。

### G3 — 真机重复验证：未开始

实体臂优先 D2、D3，各至少 3 次安全 calibration 与重复 rollout；预算为 0/5/10/25 条 transition，禁止力传感器假设。dry-run 不算真机证据。

### G4 — 论文与投稿：未完成

需完成方法图、matched 表、消融表、真实臂视频/曲线、匿名 PDF 与投稿合规检查。

## 6. 可复用与不需重跑

直接复用：MuJoCo cache、Z32/shared checkpoint、Z69 robot初始化、解析投影、Z65 posterior、Z70 rank-8 adapter、Z75安全链、bridge aligner、五seed checkpoint及冻结raw rows。

不需再训练：seed7/17/27/37/47 matched-adapter G2-R。投稿前只需按最终脚本复算汇总/图表、完成真机任务和最终测试。

只在新增真机或明确改变模型/协议时重跑：真机 D2/D3；若改变 K 规则，必须使用全新 confirmation seeds，不能把 seed17/27 重新包装为未见验证集。

## 7. 近期执行清单

1. 锁定本文件与最终报告，生成论文主表、曲线和方法图。
2. 运行实体 ST3215 + 手眼视觉 dry-run，再执行 D2/D3 低幅安全校准。
3. 完成至少三次真机重复 rollout；记录视觉时间戳、电流、温度、锁定漂移和急停事件。
4. 将真机结果与仿真 matched protocol 对齐；若未达到容差，报告失败边界，不回退到旧机制。
5. 完成 ICRA 论文与匿名合规检查；文中只使用本文件允许主张。

## 8. 复现入口与版本治理

- 权威报告：`reports/icra-senior-review-remediation-20260824.md`
- 权威汇总：`runs/g2_r0_icra_audit_20260824/summary.json`
- 原始窗口行：`runs/g2_r0_icra_audit_20260824/seed*/raw_window_metrics_30traj.json`
- 测试命令：`pytest -q`（当前 248 passed）
- 改变模型、协议、K 规则或 baseline 时，必须新建报告与新 confirmation seeds，不覆盖冻结 artifact。

**一句话状态**：IPWM在五seed、扩大轨迹与成对区间下稳定改善损伤域object指标并精确满足锁定约束，但存在一个free-joint门失败；尚缺真机pushing，不能宣称完整安全恢复或投稿就绪。
