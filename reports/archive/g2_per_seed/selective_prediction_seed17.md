# Selective Prediction Experiment

**Seed**: 17  **Domain**: D3__mixed_composition

## Setup

Ensemble disagreement (epistemic uncertainty = member std) is used as a
rejection score. For each coverage fraction, the lowest-uncertainty samples
are retained and RMSE is computed on that subset.

## Key Metrics

- Global uncertainty-error Spearman: **+0.9578**
- Coverage-RMSE Spearman (monotonicity check): **+1.0000**
- Is monotone (> 0.9): **Yes**
- Baseline RMSE (100% coverage): **0.4179**

## Selective Prediction Curve

| Coverage | N Retained | RMSE | RMSE Reduction |
|---:|---:|---:|---:|
| 100% | 450 | 0.4179 | +0.00% |
| 90% | 405 | 0.3615 | +13.50% |
| 80% | 360 | 0.3227 | +22.77% |
| 70% | 315 | 0.2824 | +32.41% |
| 60% | 270 | 0.2413 | +42.27% |
| 50% | 225 | 0.2011 | +51.86% |
| 40% | 180 | 0.1524 | +63.53% |
| 30% | 135 | 0.1092 | +73.87% |
| 20% | 90 | 0.0709 | +83.04% |

## Interpretation

If RMSE drops monotonically as coverage decreases, the ensemble uncertainty
successfully identifies hard predictions. At 50% coverage the RMSE reduction
quantifies the practical gain from selective prediction in a deployment context.

A non-monotone curve suggests uncertainty is not well-calibrated at the
sample level, even if global Spearman is high (depth-index confound).