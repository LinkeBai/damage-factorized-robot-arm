目前阶段
我们现在处于：
G0 已完成，G1 正在重新验证，尚未正式通过，暂时不能进入 G2。

更直白地说：
- 机械臂建模、基础标定、MuJoCo 环境等 G0 工作已经完成。
- Reach 路线经过五个 seed 验证后失败，不能再作为论文核心证据。
- 目前已经转向更适合验证 DFWM 的 Push 任务。
- Push 初步结果是好的，但证据还不完整，所以 G1 只能算“发现了新方向”，不能算正式完成。
现在已有的成果
- MuJoCo Reach 环境：完成。
- MuJoCo Push 环境：完成，100 步测试通过。
- DFWM、topology-only、history encoder、parameter-matched、monolithic、residual-only：代码基本具备。
- K=0/1/2/5 校准机制：已有。
- Reach 五 seed：完成，证明原先的领先是过拟合/随机性，不显著。
- Push 五 seed 的初步汇总：记录了 DFWM 多步误差改善 15.8%。
- 自动分析和 bootstrap 脚本：已有。
- 当前代码测试：114 passed。
还差什么
Push 的 15.8% 目前只有学长写下来的汇总：
- DFWM：0.1589
- topology-only：0.1888
- 相对改善：约 15.8%
但 backup 中没有对应的 Push 逐 seed CSV，也没有 Push 的完整 runs/。因此现在无法独立检查：
- 是否每个 seed 都改善；
- 是否至少 3/5 或 4/5 seed 改善；
- 95% 置信区间是否跨 0；
- 六种方法是否训练预算、参数量、数据量公平；
- 有没有某个异常 seed 拉高了平均值。
所以现阶段准确结论是：
Push 上观察到了有希望的 15.8% 信号，但尚未完成统计验证。

Backup 是什么
damage-factorized-robot-arm_experiment_backup_20260814.tar.gz 是学长在 8 月 14 日打包的实验增量备份。
它不是另一套项目，也不是你以后开发的新目录。里面主要有：
- 学长修改过的代码；
- Push 仿真模型；
- 新增实验脚本；
- Reach 五 seed 数据；
- V7 结果；
- 论文草稿；
- 实验日志；
- 一部分历史运行记录。
我已经把有价值的部分合并到当前仓库分支：
agent/import-senior-push-experiments
并提交到 GitHub PR #1。
backup 现在的作用只是：
作为学长原始资料的离线存档和追溯证据。

以后继续开发仍然使用：
C:\Users\asus\Desktop\damage-factorized-robot-arm
下一步
现在只需集中做一件事：
1. 在 Push 上重新运行完整六方法。
2. 固定相同训练数据、网络规模、训练轮数和评估轨迹。
3. 跑 seeds：7,17,27,42,51。
4. 保存每个 seed 的 one-step 和 multi-step 原始结果。
5. 运行配对 bootstrap，计算 95% CI。
6. 检查 DFWM 是否至少在多数 seed 上优于 topology-only。
7. 如果优势稳定且 CI 不跨 0，判定 G1 Go，进入 G2。
8. 如果不稳定，按照计划 Pivot，不能继续宣称“稳定提升 15.8%”。
因此现在论文最稳妥的表述是：
Reach 揭示了简单任务无法识别连续损伤程度；Push 初步表明接触动力学可能让因子化损伤建模产生价值，但该结论仍需完整多随机种子统计验证。
星期二18:40




你来进行吧
已经全部跑完并同步到 GitHub：PR #1。
正式结果
- 6 种方法
- 5 个 seeds
- D2、D3
- K=0/1/2/5
- 共 240 行结果
- 114 项测试通过
K=5 多步误差：
- DFWM：0.1589
- topology-only：0.1888
- 平均改善：15.8%
- DFWM 在 3/5 seeds 胜出
- 95% CI：[-0.0049, 0.0731]，跨 0，不显著
更关键的是：
- DFWM K=0：0.1573
- DFWM K=5：0.1589
校准数据增加后反而略差。因此 15.8% 的优势不是“少量试运行学会损坏程度”带来的，而是模型本身与 topology-only 的结构/训练差异。
闸门结论
当前 G1 对原始 few-shot residual calibration 主张判定为：
No-Go / Pivot，不能宣称 G1 已通过。

但结果并非毫无价值：它支持“结构化模型可能比 topology-only 更强”的 zero-shot 方向。下一步应选择：
1. 转向 robust zero-shot structured dynamics；或
2. 重新设计 calibration 数据和目标，让 K 真正包含可识别的损伤信息，再重跑 G1。
正式报告已写入 [g1-push-formal-gate-20260818.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-push-formal-gate-20260818.md)，原始数据也已进入 results/final/




robust zero-shot structured dynamics这个啥意思
**大白话解释**

`robust zero-shot structured dynamics` 可以拆开看：

- **robust**：面对不同故障、摩擦和电机衰减，模型仍比较稳定。
- **zero-shot**：部署到新的故障机械臂后，**不做额外校准、不更新参数**，直接使用，即 `K=0`。
- **structured dynamics**：不把所有故障混成一个黑箱变量，而是明确告诉模型“哪个关节坏了”，用结构化的损伤信息预测机械臂动力学。

合起来就是：

> **只要告诉模型哪个关节坏了，它不需要先试运行学习损坏程度，也能直接预测故障机械臂的运动。**

**和原方案的区别**

原来的 DFWM 主张是：

> 知道哪里坏了，再通过 K 条试运行数据推断“坏得多严重”，预测会越来越准。

但实验结果是：

| 校准数据 | DFWM 多步误差 |
|---|---:|
| K=0 | 0.1573 |
| K=1 | 0.1579 |
| K=2 | 0.1585 |
| K=5 | 0.1589 |

K 增加并没有帮助。因此不能再说“少样本校准有效”。

但 K=0 的 DFWM 已经比 topology-only 好：

- DFWM K=0：`0.1573`
- topology-only：`0.1888`

这就产生了新的研究问题：

> 一个显式编码机械臂故障结构的世界模型，是否能在没有目标机械臂校准数据时，对未见过的故障组合保持更好的预测和控制能力？

**不过目前还不能直接这样写论文**

因为现在的 K=0 DFWM 仍带有 residual 通道，只是输入设为零。它和 topology-only 的网络结构、参数量及训练方式可能不同，所以领先不一定来自“结构化建模”。

需要补齐公平实验：

1. 参数量严格匹配。
2. 相同训练数据、训练轮数和优化器。
3. 测试未见过的 D2/D3/D4 与 residual 组合。
4. 至少 5 seeds，最好更多。
5. 比较多步预测、分布外稳定性和 Push 控制效果。
6. 证明优势不是某个 seed 或更大网络造成的。
7. 检查 parameter-matched、monolithic 和 residual-only。

目前 parameter-matched 的多步误差是 `0.1631`，residual-only 是 `0.1614`，都非常接近 DFWM 的 `0.1589`，且差异不显著。因此 zero-shot 方向目前只是一个**值得验证的 Pivot**，还不是已经成立的新结论。

一句最准确的项目定位是：

> 少样本损伤程度推断暂未成功；下一步研究结构化损伤表示能否提升未知故障下的零样本动力学预测与控制鲁棒性。




一般来说K增加误差会变小的吧？存在过拟合？（嗯嗯，一定先验证一下逻辑有没有疏漏，这个反馈已经比较简洁了，就是一开始的那个reach的任务太简单，没过多seed验证，DeepSeek认为过拟合，然后在push任务上成功了，但15.8%似乎有点太高，一定要确认一下对比公平，任务设置也没问题）
一般预期是：**如果校准机制真的有效，K 增加后测试误差总体应该下降**。不要求每一步严格单调，但至少应看到稳定趋势，例如 K=0 > K=1 > K=2 ≈ K=5。

我们现在却是：

| K | DFWM 多步 RMSE |
|---|---:|
| 0 | **0.1573** |
| 1 | 0.1579 |
| 2 | 0.1585 |
| 5 | 0.1589 |

变化幅度很小，但方向相反。因此目前不能说“更多校准数据导致严重过拟合”，更准确的判断是：

> **residual latent optimization 没有提取出有效损伤信息，更新反而引入了轻微扰动。**

可能原因包括：

1. 世界模型训练时没有真正学会使用 residual latent。
2. 校准轨迹的随机动作没有充分激发摩擦、松动、延迟等差异。
3. latent optimization 优化的是校准集预测误差，但得到的 latent 不能泛化到 evaluation 轨迹。
4. K 增加只是加入更多相似轨迹，没有增加损伤可辨识信息。
5. 优化步数、学习率或损失函数不合适。
6. DFWM K=0 本身结构更强，所以已经领先；后续校准只是把它从较好的初始点推偏。

**15.8% 到底有没有问题**

这个数字计算没有错：

\[
(0.1888-0.1589)/0.1888\approx15.8\%
\]

但它容易被错误解释。

它只能说明：

> K=5 的 DFWM 平均误差比 topology-only 低 15.8%。

它不能说明：

> 少量校准数据让 DFWM 学会了损坏程度，因此提高了 15.8%。

因为 DFWM 在完全不校准的 K=0 时已经是 `0.1573`，甚至比 K=5 更好。这个优势不是 K 带来的。

而且公平性检查后还有几个问题：

- 只在 `3/5` seeds 上优于 topology-only。
- 95% CI 跨 0，统计上不稳定。
- 对 parameter-matched、residual-only 等强基线没有显著优势。
- 当前 Push 使用 Reach split 的元数据。
- 数据是 Push 场景里的随机动作激励，不是完整的目标导向推方块控制任务。
- 没有保存每条轨迹的接触率和方块位移，尚不能证明所有数据真的发生了有效接触。

所以学长担心“15.8% 太高，要检查公平性和任务设置”是完全正确的。我们检查后的结论是：

> **15.8% 数值可以复现，但原来的因果解释不成立，Push 实验协议也还需要加强。**

**下一步怎么判断是不是过拟合**

需要做一个专门的 calibration 诊断，而不是马上重新跑大实验：

1. 记录 K=0/1/2/5 的 calibration loss 和 evaluation loss。
2. 如果 calibration loss 下降、evaluation loss上升，就是典型的 latent 过拟合。
3. 如果两者都不下降，说明 residual latent 没学到有效信息。
4. 检查 latent norm 是否随 K 异常增大。
5. 保存方块位移、接触步数、接触力，确认校准轨迹确实激发了 residual physics。
6. 用独立 calibration/evaluation 动作分布测试泛化。
7. 加入 early stopping、latent 正则和 validation-selected learning rate。
8. 再看 K 增加是否带来一致改善。

因此现在最合理的判断不是“Reach 过拟合、Push 成功”，而是：

> **Reach 的早期领先被多 seed 推翻；Push 的 15.8% 平均差异得到复现，但 few-shot calibration 机制仍未成功，是否存在 latent 过拟合需要通过校准损失与测试损失的联合诊断确认。**




residual latent 没学到有效信息；怎么解决
核心不是直接调大学习率或增加 K，而是先让校准轨迹真正包含“坏得多严重”的可辨识信息。

**最可能的问题**

现在 Push 数据由随机关节动作生成。很多轨迹可能：

- 没接触方块；
- 接触时间太短；
- 动作幅度不足；
- 不同 residual 下产生的状态几乎一样；
- K 条轨迹高度相似。

这种情况下，latent 没有信息可学。

**建议按这个顺序解决**

1. **先做可辨识性检查**

对同一初始状态和相同动作，分别运行：

- nominal
- weak motor
- high damping
- delay
- backlash
- mixed residual

比较轨迹差异：

- 关节位置/速度；
- 方块位移；
- 接触步数；
- 接触力；
- 动作到响应延迟。

如果不同 residual 的轨迹差异小于随机噪声，模型不可能识别。

2. **重新设计校准动作**

不要继续使用纯随机动作。建立固定的主动诊断序列：

- 单关节正反向扫动：识别 backlash；
- 快速阶跃：识别 delay 和 motor strength；
- 慢速持续推动：识别 damping 和 friction；
- 固定接触方向推块：保证发生接触；
- 每个序列从相同初始状态开始。

K 不再表示“随机拿 K 条轨迹”，而是表示加入 K 种互补诊断动作。

3. **确保训练时模型使用 latent**

训练中加入辅助监督：

- 用 latent 预测 actuator strength、damping、delay、backlash；
- 或预测 residual profile 分类；
- 对相同 topology、不同 residual 添加 latent contrastive loss；
- 对相同 residual、不同轨迹约束 latent 接近。

可以使用：

\[
L=L_{prediction}+\lambda_1L_{residual}+\lambda_2L_{contrastive}
\]

这样避免 world model 完全忽略 latent。

4. **限制部署时的 latent 优化**

当前 latent optimization 可能把 K 条校准轨迹拟合过头。加入：

- `L2` 正则，让 latent 不要离零点太远；
- early stopping；
- validation trajectory；
- 更小学习率；
- 更少优化步数；
- latent 范数裁剪。

不要用同一批轨迹既优化 latent 又证明适应成功。

5. **让 K 真正递增**

固定嵌套数据：

- K=1 使用轨迹 A；
- K=2 使用 A+B；
- K=5 使用 A+B+C+D+E。

同时保证 A–E 分别覆盖阶跃、换向、持续接触等不同激励。这样才能解释 K 曲线。

6. **先跑小型诊断，不急着重新跑五 seeds**

先用 seed 7、D2、D3 做验证，要求：

- calibration loss 下降；
- 独立 evaluation loss 同时下降；
- K=5 优于 K=0；
- latent 能区分不同 residual；
- 打乱 residual 标签后优势消失。

只有这些通过，再跑完整五 seeds。

**推荐的最小修复路线**

第一轮先实现三件事：

1. 保存接触率、方块位移和 residual 间轨迹距离；
2. 用固定主动诊断动作替代随机动作；
3. 给 latent 添加 residual 参数回归辅助损失和 L2 正则。

预期通过标准：

- K=5 相对 K=0 多步误差下降至少 5%；
- D2、D3 方向一致；
- 至少 2/3 小规模 seeds 改善；
- calibration loss 和独立 evaluation loss同步下降；
- shuffled calibration 数据不能产生相同收益。

如果这些仍不通过，就说明当前 residual 信息对于任务不可辨识，应正式转向 zero-shot，而不是继续调参。




当前进度：

`[██████████████░░░░░░] 70%`

已完成：

- 找到原 K 无效原因：随机校准轨迹信息不足。
- 实现主动阶跃、换向、周期探针。
- 实现真实 `tool–block` 接触检测。
- 可辨识性诊断完成：不同 residual 的轨迹差异明显。
- `115` 项测试通过。
- 正式五 seed 原始实验已完成并保存。
- seed 7 主动校准小实验已完成。

seed 7 修复结果：

| K | DFWM 多步误差 |
|---|---:|
| 0 | 0.1918 |
| 1 | 0.3041 |
| 2 | 0.3015 |
| 5 | 0.3034 |

主动校准暂时失败。原因已经定位：

- latent norm 从 `0` 突然增至约 `5.75`；
- 校准探针与测试轨迹分布差异过大；
- `lr=0.1` 太高；
- `L2=0.001` 约束太弱；
- latent 在校准数据上过拟合。

我已经停止 seed 17/27，避免继续浪费约一小时。

接下来剩余工作：

`[ ]` 将 latent 学习率降至约 `0.01–0.02`  
`[ ]` 将 L2 正则提高到约 `0.05–0.1`  
`[ ]` 加 latent norm 限制和 early stopping  
`[ ]` 训练集加入主动探针，消除校准/测试分布断层  
`[ ]` 重新跑 seed 7 小闸门  
`[ ]` seed 7 的 K 曲线下降后再跑 3 seeds

目前程序没有卡住，也没有实验在后台运行；现在处于第二轮修复前的明确停靠点。
我不明白现在遇到什么问题了
**一句话解释**

我们希望模型看几段机械臂试运行，就能判断“机械臂坏得多严重”。

但现在模型看完试运行后，**判断错了，而且把原本还不错的预测改坏了**。

**原来是什么问题**

最开始使用随机动作作为试运行：

- K=0：`0.1573`
- K=5：`0.1589`

多看数据没有帮助，但也没有严重恶化。

说明随机动作提供的故障信息太少，模型基本没学到东西。

**改进后又出现什么问题**

我把随机动作换成更明显的测试动作，例如：

- 快速正反转；
- 阶跃动作；
- 单关节扫动；
- 周期性动作。

这些动作确实能暴露电机衰减、阻尼和延迟。

但结果变成：

| K | 误差 |
|---|---:|
| 不校准 | 0.1918 |
| 看1段试运行 | 0.3041 |
| 看2段试运行 | 0.3015 |
| 看5段试运行 | 0.3034 |

模型看了数据之后，误差大幅增加。

**为什么会这样**

训练模型时，它主要看的是较平缓的随机运动。

部署校准时，我们突然给它快速阶跃和换向动作。模型不熟悉这种数据，却强行根据它修改内部的 residual latent。

可以类比为：

> 平时只让学生看普通行驶，考试前突然让他根据漂移测试判断汽车故障。测试确实包含故障信息，但学生没学过怎样解释这种测试，于是得出了错误结论。

技术表现是：

- latent 正常初始值为 `0`；
- 校准后 latent norm 冲到约 `5.75`；
- 更新幅度过大；
- 模型被校准数据带偏；
- 独立测试误差上升。

**这是哪一种问题**

这是比较明确的：

> **校准分布不匹配 + latent optimization 过拟合。**

不是 GPU 不够，也不是 MuJoCo 运行失败，更不是数据越多必然越差。

**现在要怎么修**

需要同时做三件事：

1. 训练阶段也加入这些阶跃、换向测试，让模型学会如何解释主动校准动作。
2. 降低 latent 更新学习率，避免一次改太多。
3. 增强正则和 early stopping，一旦独立 validation 误差不再下降就停止更新。

目前状态可以概括成：

> 我们已经让故障信息“能被测出来”，但模型还不会正确使用这些信息。

所以 G1 现在仍未通过，但问题已经从“校准数据没有信息”缩小成“模型对主动校准数据的适配方式不正确”。




好的，怎么解决
按下面顺序解决，暂时不要继续跑五 seeds。

**第一步：限制 latent 不要乱跑**

当前 latent norm 从 `0` 冲到 `5.75`，更新过猛。

修改为：

- 学习率：`0.1 → 0.01`
- L2 正则：`0.001 → 0.1`
- latent 范围：`[-5,5] → [-1,1]`
- 优化步数：`50 → 20`
- 梯度裁剪：`1.0`

目标是先保证校准不会把 K=0 的预测破坏掉。

**第二步：训练时加入主动探针**

现在模型训练看随机运动，校准却看阶跃和换向，存在分布差异。

训练集改成：

- 50% 原随机轨迹；
- 25% 单关节阶跃；
- 25% 换向、周期动作。

六种方法使用完全相同的数据，保持对比公平。

**第三步：增加校准 validation**

每个 K 的数据拆为：

- K 条轨迹优化 latent；
- 1 条独立探针轨迹用于 early stopping；
- 原 evaluation 轨迹只做最终测试。

每次更新后：

- calibration loss 下降；
- validation loss也下降：保留更新；
- calibration loss下降但 validation loss上升：停止并回退；
- 最差情况下回退到 `z=0`，保证校准不会比 K=0 更差。

**第四步：记录诊断数据**

每次校准保存：

- calibration loss；
- validation loss；
- evaluation RMSE；
- latent norm；
- 优化步数；
- K；
- 接触步数和方块位移。

这样可以明确区分“没学到”和“过拟合”。

**第五步：只跑 seed 7**

通过条件：

- K=1 不明显差于 K=0；
- K=2、K=5 总体下降；
- K=5 相对 K=0 至少改善 5%；
- latent norm 不超过 1；
- validation 和 evaluation 方向一致。

seed 7 不通过就继续诊断，不跑其他 seeds。

**第六步：扩展到 3 seeds**

seed 7 通过后跑 `7/17/27`：

- 至少 `2/3` seeds 改善；
- D2、D3 方向一致；
- K 增加总体改善；
- shuffled calibration 不产生相同收益。

通过后才恢复正式五 seeds。

最核心的工程原则是：

> 校准只能在有独立证据证明有效时应用；否则自动回退到 K=0。

这样即使 residual latent 暂时学不好，也不会把原本的预测结果改坏。





好的，按你说的来
已按方案完成修复和 seed 7 验证。

**修复效果**

| K | 修复前 | 修复后 |
|---|---:|---:|
| 0 | 0.1918 | 0.23828 |
| 1 | 0.3041 | 0.23545 |
| 2 | 0.3015 | 0.23529 |
| 5 | 0.3034 | 0.23524 |

- latent norm：`5.75 → 约 0.52`
- K 曲线恢复单调下降。
- K=5 比 K=0 改善 `1.28%`。
- 115 项测试通过。
- 已加入主动探针、独立 validation、early stopping、强正则、限幅和失败回退。

**结论**

过拟合失控的问题解决了，但校准收益只有 `1.28%`，没有达到预设的 `5%`。因此我没有继续跑 17/27，避免把微弱信号放大包装。

现在科学上的问题是：

> 模型已经会正确使用少量校准信息，但 residual 信息带来的增益太小，不足以支撑论文核心创新。

代码和报告已同步到 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。下一步应选择“增加 residual 辅助监督/对比学习”或正式转 zero-shot，而不是继续盲目增加 seeds。





学长的提示也是这么给的吗？
不完全一样。

学长给出的核心提示是：

> Reach 可能因为任务简单和 seed 太少产生过拟合；Push 的 15.8% 看起来偏高，需要检查对比是否公平、任务设置是否正确。

也就是说，学长要求的是**审计和验证**，并没有直接提出：

- 主动校准探针；
- latent 强正则；
- early stopping；
- K=5 至少改善 5%；
- 转向 zero-shot。

这些是我根据学长的提醒，检查代码和重新跑实验后提出的工程解决方案。

学长的担心后来被验证为合理：

1. `15.8%` 数值可以复现。
2. 但只在 `3/5` seeds 上改善。
3. 95% CI 跨 0。
4. DFWM 从 K=0 到 K=5 没有改善。
5. 当前 Push 实际是随机动作下的接触动力学预测，不是完整推方块控制任务。
6. D3 的主动探针没有发生工具与方块接触。
7. 因而不能把 15.8% 解释为“少样本校准成功”。

所以可以这样向学长汇报：

> 按您的建议，我们复核了 15.8% 的公平性和任务设置。数值能够复现，但统计置信区间跨 0，且 K 增加没有改善，说明优势不是 few-shot calibration 带来的。进一步加入主动探针和正则后，K 曲线恢复下降，但 seed 7 仅改善 1.28%，未过 5% 小闸门，因此暂未继续多 seed。当前建议补 residual 辅助监督，或者将论文主张转为 zero-shot structured dynamics。

因此不是“学长也给了完全相同的解决方案”，而是：

> **学长指出了需要怀疑和复核的地方；我们通过实验确认了问题，并继续定位到 residual latent 校准机制过弱。**





那15.8%不是显著提升吗？学长工作的意义在哪里？
15.8% 是一个**幅度不小的平均提升**，所以学长的工作当然有意义。问题在于：

> **效果量大，不等于统计显著，也不等于证明了原机制。**

**15.8%说明了什么**

学长成功发现：

- Reach 太简单，无法体现连续故障程度；
- Push 的接触动力学更能放大电机衰减、阻尼、延迟等影响；
- DFWM 在 Push 上的平均多步误差为 `0.1589`；
- topology-only 为 `0.1888`；
- 平均相对改善 `15.8%`。

这说明 Push 确实比 Reach 更有研究价值，也证明模型结构可能包含有效信息。

**为什么还不能叫“显著提升”**

五个 seed 的结果方向并不完全一致：

- seed 7：DFWM 更好
- seed 17：DFWM 更好
- seed 27：DFWM 更差
- seed 42：DFWM 更好
- seed 51：DFWM 更差

即 `3/5` 改善。

而且 95% CI 为：

`[-0.0049, 0.0731]`

它跨过 0，意味着根据当前样本，还不能排除“换一组 seeds 后优势消失”的可能。

因此准确说法是：

> DFWM 观察到 15.8% 的平均改善，效果量较大，但在五 seed 下尚未达到统计显著。

不能写成：

> DFWM 稳定、显著地提升了 15.8%。

**更关键的是机制解释**

原主张是：

> 模型通过 K 条试运行推断“坏得多严重”，因此提高预测精度。

但实际 DFWM：

- K=0：`0.1573`
- K=5：`0.1589`

完全不做校准反而更好。因此 15.8% 不能归功于 few-shot calibration。

它更可能来自：

- DFWM 网络结构；
- residual 通道参与训练带来的表征能力；
- 网络容量差异；
- 优化过程差异；
- 某些 seeds，特别是 seed 42，拉高了平均改善。

**学长工作的真正意义**

学长至少完成了四个非常关键的贡献：

1. **推翻错误的 Reach 结论**

   发现早期领先属于小 seed/过拟合假象，避免论文建立在错误结论上。

2. **找到更合适的任务**

   Push 比 Reach 更能体现连续动力学残差，这是重要的任务设计进展。

3. **发现一个有潜力的新信号**

   15.8% 即使暂不显著，也值得继续研究。没有这个结果，我们甚至不知道该往哪里改。

4. **暴露原机制的真实问题**

   重新分析后发现，优势不是 K 校准带来的。这帮助我们把问题从“模型完全没用”缩小到：

   > 模型结构可能有效，但 residual inference 机制尚未有效。

所以学长的工作不是“结果作废”，而是完成了一次关键的研究转折：

> **他证明了 Push 是正确方向，并发现了结构优势信号；我们进一步确认，这个信号目前还不能被解释成 few-shot residual calibration 成功。**

最适合发给学长的一句话是：

> 15.8% 的平均提升已成功复现，说明您将任务从 Reach 转向 Push 是有效的；进一步审计发现该优势在 3/5 seeds 上成立但 CI 跨 0，而且 K=0 已经有同等优势，因此当前更像结构/zero-shot 收益，而不是 few-shot calibration 收益。下一步应保留 Push 主线，重点修 residual inference 和公平性验证，而不是否定已有工作。




那我们现在应该怎么推进         ？
现在不要继续盲目调 latent，也不要放弃 Push。建议采用“主线保成果，支线修机制”。

**主线：Zero-shot structured dynamics**

把学长发现的 15.8% 作为待验证的结构优势，先回答：

> 不做目标故障校准时，结构化故障建模是否比 topology-only 更稳健？

立即补：

1. 参数量严格匹配。
2. 统一训练数据、epoch、优化器和 early stopping。
3. 使用 K=0，不做 latent optimization。
4. D2、D3，后续加入 D4 和 `mixed_unseen`。
5. 运行 5–10 seeds。
6. 报告多步误差、逐 seed 结果、bootstrap CI。
7. 增加 shuffled topology 和参数量匹配消融。

如果结构化模型仍稳定领先，论文主张可以改为：

> 面向未知关节故障组合的结构化零样本动力学泛化。

这是目前最稳、离现有结果最近的路线。

**支线：Few-shot calibration**

保留但降级为增强实验，不再作为当前主结论。

现在已经做到：

- 主动探针能区分 residual；
- latent 不再爆炸；
- K 曲线恢复下降；
- seed 7 改善 `1.28%`。

下一步需要加入：

- residual 参数回归辅助损失；
- 同 residual 不同轨迹的 latent 一致性损失；
- 不同 residual 的对比损失；
- shuffled calibration 负对照。

只有单 seed 改善超过 5%，才重新跑 3 seeds。

**Push 任务必须重新规范**

目前最大的任务风险是“场景里有方块”不等于真正完成 Push。

需要建立 Push 专用协议：

1. 固定方块初始位置和目标区域。
2. 保存工具与方块接触步数。
3. 保存方块位移、最终距离和成功率。
4. 确保 D2、D3 都存在可推动目标。
5. 校准轨迹和评估轨迹严格分离。
6. 使用 Push 专用 split，不能继续标记为 `task: reach`。
7. 将“动力学预测”和“推块控制”分成两张结果表。

如果 D3 在当前锁定角无法接触方块，应调整方块位置或筛选 D2/D3 共同可达的 Push 工作区，而不是直接加高方块或忽略无接触。

**推荐执行顺序**

`[1]` 修正 Push 专用任务、共同可达目标和接触统计  
`[2]` 跑公平的 K=0 zero-shot 三 seed smoke  
`[3]` 有一致优势后扩展到 5–10 seeds  
`[4]` 加 D4、mixed_unseen 和参数量消融  
`[5]` 再处理 residual auxiliary/contrastive learning  
`[6]` few-shot 单 seed 超过 5% 后才恢复多 seed  
`[7]` 最后加入控制实验和论文主表

**当前论文定位**

暂时写成：

> 我们发现简单 Reach 无法揭示连续损伤差异，而接触丰富的 Push 动力学暴露了结构化故障建模的潜在优势。当前证据更支持 zero-shot 结构泛化；few-shot residual calibration 仍作为待验证扩展。

所以最优决策是：

> **保留学长的 Push 路线，把 zero-shot 做成可靠主线；few-shot 不删除，但降为支线，过小闸门后再升级。**




那按照你的来吧
已按路线推进完，结果比预期更关键。

**完成内容**

- 建立 Push 专用 split，不再沿用 Reach 配置。
- 找到 D2/D3 共同接触工作区。
- 使用“先接近方块、再推向目标”的真实 Push 轨迹。
- 精确记录工具与方块接触次数、方块位移。
- 完成 K=0 zero-shot smoke。
- 115 项测试通过，结果已同步到 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。

**Zero-shot 结果**

| Seed | DFWM | topology-only |
|---|---:|---:|
| 7 | 0.5746 | **0.4966** |
| 17 | 0.5057 | **0.4273** |

两个 seed 都是 DFWM 更差。即使 seed 27 获胜，也无法通过 `2/3` 闸门，所以已停止。

**当前科学结论**

原来的 `15.8%` 是在“Push 场景中的随机关节运动预测”上得到的；换成真正发生接触和推块的目标导向 Push 后，优势消失并反转。

因此：

- Few-shot calibration：No-Go，修复后只有 `1.28%`。
- Robust zero-shot：No-Go，规范 Push 下连续两个 seed 失败。
- 原 `15.8%`：可以保留为诊断结果，但不能作为论文主结果。
- 当前 G1：仍未通过，不能进入正式 G2。

下一步已经不是调几个超参数，而是方法重构：

> 在训练时加入 residual 参数监督和对比学习，让 latent 明确编码电机衰减、阻尼、延迟与 backlash，再重新跑单 seed 机制闸门。

正式报告在 [g1-push-zero-shot-smoke-20260818.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-push-zero-shot-smoke-20260818.md)。




按你说继续，不要中断，可以跑大概10h，我要睡了，记住：一直跑
连续实验已全部完成，没有中途停在单个失败配置。

**最终结果**

- residual embedding 监督权重 `0.1/1/5`：全部失败，K 增加无改善。
- history encoder 监督权重 `1/10`：失败。
- 逐轨迹 history 监督：失败。
- residual-only：
  - seed 7：改善 `6.56%`
  - seed 17：恶化 `5.52%`
  - seed 27：恶化 `0.75%`
  - 只有 `1/3` seeds 改善，未过闸门。
- 规范目标导向 Push 下，DFWM、zero-shot、few-shot、history 和 residual-only 均未形成稳定优势。

**结论**

> 当前 DFWM 实现的 G1 正式判定为 **No-Go**，暂停 G2。

原 `15.8%` 只能说明随机关节激励数据上的平均差异，不能证明 few-shot recovery，也不能证明真正的目标导向 Push 性能。

**已完成归档**

- 115 项测试通过。
- 所有关键 CSV 已放入 `results/final/`。
- Plan V5、实验日志和 README 已更新。
- 最终审计报告：[g1-overnight-method-audit-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-overnight-method-audit-20260819.md)
- GitHub 已同步：[PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)

下一步不能再靠调参，应在“独立系统辨识模块 + residual dynamics correction”与“失败边界/benchmark 论文 Pivot”之间做研究方向选择。




那则么办
先别慌。现在不是论文做不下去，而是原假设被严格验证为不成立，必须换方法，不能继续给旧模型调参。

**下一步只做一个关键实验：Oracle 上界测试**

直接把仿真中真实的 residual 参数告诉模型，不让模型自己猜。

它回答一个决定性问题：

> 如果模型知道机械臂到底坏得多严重，预测能不能明显改善？

会有两种结果：

1. **Oracle 明显改善，例如超过 10%**

   说明世界模型能够利用 residual，问题只出在“怎么从试运行识别 residual”。

   下一步做独立系统辨识器：

   `主动探针 → residual 参数估计 → 冻结动力学模型`

   这仍然有论文空间，而且逻辑更干净。

2. **Oracle 也不改善**

   说明当前 world model 结构根本不会利用 residual。

   这时停止 DFWM，改成显式 residual correction：

   \[
   \hat{s}_{t+1}=f_{\text{nominal}}(s_t,a_t,\text{topology})
   +g_{\text{residual}}(s_t,a_t,z)
   \]

   即基础模型预测正常部分，小修正网络只学习故障造成的差值。

**执行顺序**

1. 实现 Oracle residual descriptor。
2. 只跑 seed 7、D2/D3、目标导向 Push。
3. 比较 topology-only、DFWM K=0、Oracle DFWM。
4. Oracle 改善超过 10%才扩展 3 seeds。
5. Oracle 不通过，立即转 residual correction，不再调 latent。
6. 把当前所有失败实验作为方法设计依据和消融证据。

**给学长的说法**

> 我们完成公平性复核后确认，原 15.8% 不能归因于 few-shot calibration。主动校准、history encoder 和 residual-only 都没有跨 seed 稳定。下一步先做 Oracle residual 上界实验，区分是 residual 不可辨识，还是 world model 本身不会利用 residual；根据结果决定做独立系统辨识器还是显式 residual correction。

现在最重要的是 Oracle 测试。它成本低，却能决定整篇论文下一步的技术路线。




好的
Oracle 实验已经完成。

| 方法 | 多步 RMSE |
|---|---:|
| topology-only | 0.5135 |
| DFWM K=0 | 0.4891 |
| DFWM + 真实 residual | **0.4748** |

真实 residual 相对：

- DFWM K=0 改善 `2.9%`
- topology-only 改善 `7.5%`
- D2、D3 方向一致

这说明：

> 模型确实能利用 residual，但能力很弱。即使直接告诉它真实损伤参数，上限也只有 2.9%，所以问题不只是“识别不准”，当前拼接式世界模型本身也不够好。

因此下一步确定为：

\[
\hat{s}_{t+1}
=
f_{\text{base}}(s_t,a_t,\text{topology})
+
g_{\text{residual}}(s_t,a_t,z)
\]

也就是把基础动力学和损伤修正明确拆成两个网络，而不是继续把 topology 和 residual 拼到一个 context 里。

结果、报告和代码已同步到 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)，115 项测试通过。正式报告是 [g1-oracle-residual-upper-bound-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-oracle-residual-upper-bound-20260819.md)。




这个还属于世界模型吗
属于世界模型，但属于更明确的**组合式世界模型**，不是给原世界模型简单套皮。

结构是：

\[
\hat{s}_{t+1}
=
f_{\text{base}}(s_t,a_t,\text{topology})
+
g_{\text{residual}}(s_t,a_t,z)
\]

其中：

- `f_base`：主要世界模型，预测机械臂在已知关节故障拓扑下的基础动力学。
- `g_residual`：残差动力学分支，只预测电机衰减、阻尼、延迟、回差等造成的额外偏差。
- 两者相加得到完整的下一状态预测，仍然可以进行多步 rollout、MPC 和策略规划。

**与原 DFWM 的区别**

原模型是：

> 把 topology 和 residual 拼接起来，交给同一个 GRU 自己分辨。

问题是模型容易忽略 residual，Oracle 真实参数也只改善 `2.9%`。

新模型是：

> topology 负责基础运动，residual 被强制负责“基础模型预测错了多少”。

所以 residual 不容易被主模型吞掉，因果分工更清楚。

**论文中可以怎么称呼**

可以叫：

- `Residual-Corrected World Model`
- `Factorized Base-and-Residual World Model`
- `Topology-Conditioned World Model with Residual Dynamics Correction`

它仍然是世界模型，因为它：

- 学习环境状态转移；
- 预测未来状态；
- 支持多步想象；
- 可以被 MPC 或 actor 用于决策。

但它不是严格意义上的 Dreamer。更准确的定位是：

> **条件动力学世界模型 + 显式残差修正。**

这也符合 Plan V5 的 Pivot 条件：当原 WM 不稳定但条件动力学仍有价值时，转为 `conditional dynamics + MPC`，不再强行使用 Dreamer 品牌。




你还在跑吗
已经跑完，不在运行了。

结果：

- `116` 项测试全部通过
- seed 7 总体多步误差改善：`7.27%`
- D2：明显改善，`0.5152 → 0.4333`
- D3：反而变差，`0.5079 → 0.5154`
- 未达到预设 `10%` 门槛，且 D2/D3 方向不一致

结论：显式残差修正仍属于世界模型，但当前结构尚未通过 G1。问题不只是 residual latent 推断，残差修正的跨故障泛化也不足。结果位于 [push_results.csv](C:/Users/asus/Desktop/damage-factorized-robot-arm/runs/g1_push_residual_correction/seed7_e40/push_results.csv)。




那如何解决
问题已经定位：**不是 K 不够，而是残差修正器在 D3 上学错了**。它只优化单步误差，却被拿去做多步滚动，误差会累积，导致 D2 有效、D3 失效。

按这个顺序解决：

1. **改为预测状态增量**
   - 预测 `Δstate`，不直接修正绝对下一状态。
   - 对关节、方块位置、速度分别归一化。

2. **加入门控**
   - 使用  
     `next_state = base_prediction + gate × residual_correction`
   - `gate` 限制在 `0~1`，避免 D3 被过度修正。
   - 修正器最后一层继续零初始化，初始行为等于 topology-only。

3. **改用多步训练**
   - 损失同时包含 `1/5/10-step rollout loss`。
   - 重点优化论文真正评估的多步预测，而不只是 teacher-forcing 单步误差。

4. **分离不同物理量**
   - actuator loss、backlash 修正机械臂关节状态；
   - friction/contact 修正方块状态；
   - 避免一个 8 维 residual 向量粗暴修正全部 14 维状态。

5. **先做 oracle 门槛实验**
   - seed 7 上要求 D2、D3 都改善；
   - 平均改善至少 `10%`；
   - 通过后再跑 seeds `7/17/27`；
   - oracle 都不通过，就不再浪费时间研究 K 和 latent inference。

这仍然是世界模型，准确名称可以定为 **damage-conditioned compositional world model**。当前最值得做的是“门控 + 分组修正 + 多步损失”，而不是继续堆 history encoder 或增加 K。




完整复验结束，这个方向淘汰：

- D2：`-1.34%`
- D3：`+1.55%`
- 总体：仅 `+0.09%`

快速筛选时的约 `5.25%` 没有在完整训练中复现，说明低保真筛选存在排序偏差，不能直接作为论文证据。

接下来不再微调这个修正器。更有价值的方向是把 residual 从“输出后补丁”改成**动力学内部调制**：

- residual 通过 FiLM/低秩 adapter 调节世界模型隐藏动力学；
- 机械臂状态和接触物体状态分别调制；
- topology 主干冻结，adapter 参数保持很少；
- 先做 `12 epochs + 3 candidates` 筛选；
- 筛选必须加入独立验证轨迹，只有 D2/D3 均改善才完整训练。

这仍保留 DFWM 的核心主张，但比“给世界模型输出套修正层”更有方法价值。当前训练已全部停止，没有后台进程。




这轮新方向已经验证完毕：

- 输出修正：完整实验不稳定，淘汰。
- 静态 FiLM：中等实验 D2/D3 均退化，淘汰。
- 动态 FiLM：D2 略有改善，但 D3 持续退化，淘汰。
- 没有浪费时间继续跑完整 40 epochs 或多 seed。
- `119` 项测试全部通过。
- 诊断记录已写入 [g1-residual-adapter-audit-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-residual-adapter-audit-20260819.md)。

当前证据说明：**继续调 residual latent、K 或 adapter 收益很低**。下一条更值得挖掘的路线应按 Plan V5 Pivot 转向 **robust zero-shot structured dynamics + uncertainty-aware control**：不再声称少量数据能识别故障严重程度，而是研究面对未知故障时，结构化世界模型如何提供更稳健的预测和控制。




已正式转向 **robust zero-shot structured dynamics**。

当前结果：

- 3 成员 topology ensemble 中等实验通过预测闸门。
- D2 相对平均单模型提升 `23.8%`。
- D3 相对平均单模型提升 `24.3%`。
- 当前模型方差在 D3 上没有可靠校准，因此不使用 uncertainty penalty。
- 已实现 minimax Push CEM planner，用各成员最坏预测代价规划。
- `121` 项测试全部通过。
- checkpoint 已保存：`runs/g1_robust_zero_shot/seed7_medium_v1/ensemble.pt`
- Pivot 记录：[g1-robust-zero-shot-pivot-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-robust-zero-shot-pivot-20260819.md)

下一道低成本闸门已经明确：用同一 checkpoint，在 D2、D3 各一个目标上比较 ensemble-mean MPC 与 minimax MPC。只有 minimax 不损害任一故障，并改善最差域的方块终点误差，才进入多目标、多 seed。




**目前看，这个方向是对的，但还没证明完整成立。**

理由很清楚：

- 原 residual/少样本校准路线连续失败，连 oracle 都没有稳定优势，继续调 K 或 latent 意义不大。
- topology ensemble 在中等保真测试中稳定改善：
  - D2：`23.8%`
  - D3：`24.3%`
- 改善在两个故障上方向一致，比之前只在某个配置或某个故障上有效可靠得多。
- 它符合计划书的 Pivot：当 factorized residual 优势不成立时，转向 robust zero-shot。

但要谨慎区分两个结论：

- **已经初步成立**：ensemble 的 zero-shot 多步预测优于平均单模型。
- **尚未成立**：这种预测优势能否转化成 Push 控制成功率或终点误差改善。

另外，不确定性估计目前并不可靠，尤其 D3。因此正确表述是：

> topology-conditioned ensemble + minimax robust planning

暂时不要表述为：

> calibrated uncertainty-aware control

下一步 MPC 闸门非常关键。若 minimax MPC 在 D2、D3 都不退化，并改善最差域，再跑多 seed，这条路线就具备成为论文主线的潜力；若控制端没有收益，则只能作为预测 benchmark 结果，不能撑起完整 ICRA 方法。




继续完成了最小控制闸门，但发现了新的基础问题：

- robust ensemble 的预测优势仍成立：
  - D2：约 `23.8%`
  - D3：约 `24.3%`
- minimax Push MPC 已实现并通过测试。
- 但闭环 Push benchmark 没有真正推动方块：
  - nominal IK 位移为 `0`
  - ensemble-mean 位移为 `0`
  - minimax 位移为 `0`
- 因此现在不能比较 MPC 方法优劣，问题在 Push 控制协议，而不是 minimax 方法本身。
- `121` 项测试全部通过。

当前正确顺序是：先让确定性的 nominal controller 在评估目标上稳定接触并推动方块，再比较 mean/minimax MPC。诊断已同步到 [g1-robust-zero-shot-pivot-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-robust-zero-shot-pivot-20260819.md)。




已完成并同步 GitHub。

关键结果：

- 修复了旧 Push 协议的零接触问题，包括夹爪碰撞几何、推送方向和目标划分。
- corrected Push 在 D2/D3 所有评估目标均有真实接触和方块位移。
- ensemble 预测在 3/3 seeds、D2/D3 上一致改善，约 `17.7%–34.7%`。
- guarded ensemble MPC：
  - seed 7：最差域误差改善 `13.3%`
  - seed 17：改善 `33.9%`
  - seed 27：改善 `2.9%`
  - 3/3 seeds 方向一致，所有控制回合保持成功。
- `121` 项测试全部通过。
- 正式报告：[g1-robust-zero-shot-corrected-results-20260819.md](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-robust-zero-shot-corrected-results-20260819.md)
- GitHub Draft PR：[PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)

当前可以判断：**修正后的 robust zero-shot Pivot 已通过 G1 最小机制闸门**，但还需要置信区间、参数匹配 ensemble baseline 和更广目标测试，才能升级为论文主结果。




继续完成并同步 GitHub。

新增结论：

- 参数匹配宽单模型参数：`460,382`
- ensemble 参数：`450,906`
- ensemble 参数更少，但预测仍显著更好：
  - 对平均成员：`24.1%`，95% CI `[18.0%, 31.0%]`
  - 对参数匹配宽模型：`36.6%`，CI `[24.8%, 44.3%]`

扩展至 5 个目标后，guarded MPC：

- seed 7：改善 `7.3%`
- seed 17：改善 `30.6%`
- seed 27：`-0.06%`
- 2/3 seeds 改善
- 11/15 目标改善
- 15/15 回合保持成功
- 控制 CI 跨 0，所以暂时只能算 G1 通过，不能作为稳定论文结论

`122` 项测试通过。正式报告已更新：[G1 corrected results](C:/Users/asus/Desktop/damage-factorized-robot-arm/reports/g1-robust-zero-shot-corrected-results-20260819.md)。

GitHub 已同步，远端提交：`5638900`。当前最强、最可靠的论文结果是 **参数匹配条件下的 robust zero-shot ensemble prediction**；guarded control 暂时作为次要结果。




已继续完成并同步 GitHub。

- 5-seed 验证全部完成，D2/D3 均为正向。
- 对参数匹配单模型平均误差降低 **30.7%**。
- 95% bootstrap CI：**[15.1%, 42.6%]**。
- seed 47 提升较弱，已如实记录，说明存在初始化方差。
- 增加训练耗时记录和参数匹配模型 checkpoint。
- 全套测试：`122 passed`。
- 已同步到 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。

当前结论：**G1 的预测机制验证已达到五随机种子最低证据标准**；控制结果仍作为次要结果，不夸大为稳定提升。




**当前阶段**
项目已完成 **G0**，并基本完成 **G1 最小机制验证**。当前论文方向已从原始 DFWM residual latent，转向：

> **Robust Zero-Shot Structured Dynamics**：通过多个拓扑条件世界模型组成集成，在未知故障强度下提高多步动力学预测的稳定性，并利用模型分歧估计不确定性。

**已经完成**
- MuJoCo 机械臂与 Push 接触模型已修正。
- 修复旧 Push 实验“没有真实接触和方块位移”的协议问题。
- D2、D3 故障条件下均产生有效接触与位移。
- 完成 3 模型集成与参数匹配单模型的公平对比。
- 完成 **5 个随机种子**：`7/17/27/37/47`。
- 模型参数量公平：
  - 集成模型：450,906
  - 参数匹配单模型：460,382
- 完成冻结模型、零样本测试，没有在线微调或数据泄漏。
- 完成 guarded MPC 初步控制验证。
- 每次实验均保存 checkpoint、结果和配置。
- 全套测试：**122 项全部通过**。
- 代码、结果和报告已同步到 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。

**核心结果**
5-seed 多步预测结果：

| 对比方法 | 平均误差改善 | 95% 置信区间 | 正向种子 |
|---|---:|---:|---:|
| 平均单个集成成员 | 20.7% | [15.3%, 26.4%] | 5/5 |
| 参数匹配单模型 | **30.7%** | **[15.1%, 42.6%]** | **5/5** |

分故障结果：

- D2 相对参数匹配模型平均改善：**30.0%**
- D3 相对参数匹配模型平均改善：**31.4%**
- 两个故障均为 **5/5 seeds 正向**

seed 47 只有约 2.1% 的平均改善，说明效果存在明显初始化方差，但总体置信区间仍不跨零。

**控制结果**
Guarded MPC 在扩大到 5 个目标后：

- 3 个 seed 中 2 个改善；
- 11/15 个 seed-target 组合改善；
- 15/15 次控制均保持成功；
- 置信区间跨零。

因此控制部分只适合作为次要机制结果，目前不能声称稳定提升。

**被否定的旧结论**
- Reach 上原先的大幅提升主要来自任务过于简单和过拟合。
- 旧 Push 的 15.8% 提升来自错误接触协议，不能用于论文。
- residual latent、FiLM adapter 没有稳定通过 D2/D3 验证。
- 这些失败已记录，后续不再作为论文主张。

**G1 判断**
**G1 的预测机制验证已经完成并通过。**

但这只证明当前机制“值得继续”，还不能直接证明达到 ICRA 论文创新强度。现在最可靠的主结果是：

> 在参数量、训练数据和测试轨迹一致的条件下，拓扑条件模型集成在未知故障强度的 Push 动力学预测中，获得稳定的 5-seed 多步预测优势。

下一阶段应进入论文级强化：增加强基线、扩大故障强度和目标分布、做消融实验，并明确“结构化条件”相对于普通深度集成的新增价值。




**结论：可以交付 G1 阶段成果，但不能宣称原计划 G1 全部完成。**

原因是项目已按计划书的 **Pivot 条款**转向 `robust zero-shot structured dynamics`。因此可以作为“G1 阶段评审包”交付，但原始 DFWM 路线中的部分固定项没有完成或已被实验否定：

- 原定 4 方法完整对比尚未全部按统一 Push 协议重跑。
- `K=0/1/2/5` passive calibration 曲线尚未形成有效主结果。
- residual latent optimization 已实现，但未学到稳定有效信息。
- actor-head/history 等原始基线尚未完成论文级统一比较。
- Guarded MPC 仅达到初步门槛，统计显著性不足。
- 当前核心创新与普通 deep ensemble、domain randomization 等强基线的区别还需验证。

**已经可以交付的内容**
- 可运行 MuJoCo Push 环境和修正后的接触协议。
- dataset generator、条件世界模型和模型集成。
- D2/D3、5 seeds、参数匹配公平实验。
- checkpoint、manifest、日志、结果表及训练耗时。
- 5-seed bootstrap 统计。
- 冻结模型的 guarded MPC 初步结果。
- 完整实验审计报告。
- `122` 项测试全部通过。
- GitHub 可复现代码与结果：[PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。

**推荐汇报口径**
> G0 已完成；G1 原始 DFWM 假设未通过，项目依照 Pivot 条款完成 robust zero-shot 路线的最小机制验证。5-seed 预测实验通过，但控制效果和创新性对比仍需在下一阶段补强。

所以：**可以交给学长复现和阶段验收，也可以结束 G1 探索；暂时不能作为论文最终实验交付。**




已完成：

- 修改 `PROJECT-PLAN-V6.md` 的 G1 交付物，按实际 Pivot 成果验收。
- 正式启动 G2，并冻结实验协议。
- 完成普通 Deep Ensemble 公平基线，5 seeds 全部跑完。
- 两种方法参数量均为 `450,906`，共享训练数据与评估轨迹。
- 全套测试：`123 passed`。
- 已同步 [GitHub PR #1](https://github.com/LinkeBai/damage-factorized-robot-arm/pull/1)。

**G2 首轮结论**
Structured Ensemble 相对普通 Deep Ensemble：

- 平均改善：**2.47%**
- 95% CI：**[-1.83%, 6.38%]**
- 4/5 seeds 正向，但 seed 47 为负

因此未通过 G2 方法门。之前的 30.7% 主要证明“集成优于宽单模型”，暂不能证明 topology structure 有独立贡献。

按照计划书，当前触发 **benchmark Pivot**。下一步应补 Domain-randomized Ensemble 完善 benchmark，或者先提出一个区别于普通集成的新机制，再重新开启方法验证。