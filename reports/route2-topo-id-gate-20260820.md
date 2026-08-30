# 路线2 Gate 报告：集成分歧指纹拓扑识别

**日期**: 20260820
**假设**: 集成成员的 per-dimension 分歧指纹携带故障拓扑信息，K 条主动探测轨迹可识别锁定关节

## 核心结果

| 指标 | 值 |
|---|---|
| K=5 平均改善 | +0.01% |
| K=5 识别准确率 | 100.0% |
| **Gate 决定** | **PARTIAL** |

## 决策理由

识别准确率 100.0% 但预测改善不显著，需进一步分析

## 各 K 值结果

| Domain | K | RMSE | vs K=0 | 识别准确率 |
|---|---:|---:|---:|---:|
| D2__mixed_composition | 0 | 0.4943 | +0.00% | N/A |
| D2__mixed_composition | 1 | 0.4942 | +0.02% | 1.0 |
| D2__mixed_composition | 2 | 0.4942 | +0.02% | 1.0 |
| D2__mixed_composition | 5 | 0.4942 | +0.02% | 1.0 |
| D3__mixed_composition | 0 | 0.4388 | +0.00% | N/A |
| D3__mixed_composition | 1 | 0.4388 | +0.00% | 1.0 |
| D3__mixed_composition | 2 | 0.4388 | +0.00% | 1.0 |
| D3__mixed_composition | 5 | 0.4388 | +0.00% | 1.0 |