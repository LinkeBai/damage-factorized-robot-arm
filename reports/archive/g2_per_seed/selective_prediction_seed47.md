# Selective Prediction Experiment

**Seed**: 47  **Domain**: D3__mixed_composition

## Setup

Ensemble disagreement (epistemic uncertainty = member std) is used as a
rejection score. For each coverage fraction, the lowest-uncertainty samples
are retained and RMSE is computed on that subset.

## Key Metrics

- Global uncertainty-error Spearman: **+0.9731**
- Coverage-RMSE Spearman (monotonicity check): **+1.0000**
- Is monotone (> 0.9): **Yes**
- Baseline RMSE (100% coverage): **0.5558**

## Selective Prediction Curve

| Coverage | N Retained | RMSE | RMSE Reduction |
|---:|---:|---:|---:|
| 100% | 450 | 0.5558 | +0.00% |
| 90% | 405 | 0.4819 | +13.30% |
| 80% | 360 | 0.4338 | +21.96% |
| 70% | 315 | 0.3838 | +30.96% |
| 60% | 270 | 0.3300 | +40.63% |
| 50% | 225 | 0.2716 | +51.13% |
| 40% | 180 | 0.2172 | +60.93% |
| 30% | 135 | 0.1551 | +72.10% |
| 20% | 90 | 0.1034 | +81.39% |

## Interpretation

If RMSE drops monotonically as coverage decreases, the ensemble uncertainty
successfully identifies hard predictions. At 50% coverage the RMSE reduction
quantifies the practical gain from selective prediction in a deployment context.

A non-monotone curve suggests uncertainty is not well-calibrated at the
sample level, even if global Spearman is high (depth-index confound).