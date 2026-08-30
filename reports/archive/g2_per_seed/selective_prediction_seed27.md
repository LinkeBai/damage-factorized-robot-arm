# Selective Prediction Experiment

**Seed**: 27  **Domain**: D3__mixed_composition

## Setup

Ensemble disagreement (epistemic uncertainty = member std) is used as a
rejection score. For each coverage fraction, the lowest-uncertainty samples
are retained and RMSE is computed on that subset.

## Key Metrics

- Global uncertainty-error Spearman: **+0.9108**
- Coverage-RMSE Spearman (monotonicity check): **+1.0000**
- Is monotone (> 0.9): **Yes**
- Baseline RMSE (100% coverage): **0.4899**

## Selective Prediction Curve

| Coverage | N Retained | RMSE | RMSE Reduction |
|---:|---:|---:|---:|
| 100% | 450 | 0.4899 | +0.00% |
| 90% | 405 | 0.4196 | +14.35% |
| 80% | 360 | 0.3787 | +22.70% |
| 70% | 315 | 0.3301 | +32.62% |
| 60% | 270 | 0.2822 | +42.40% |
| 50% | 225 | 0.2331 | +52.41% |
| 40% | 180 | 0.1924 | +60.73% |
| 30% | 135 | 0.1507 | +69.24% |
| 20% | 90 | 0.1067 | +78.23% |

## Interpretation

If RMSE drops monotonically as coverage decreases, the ensemble uncertainty
successfully identifies hard predictions. At 50% coverage the RMSE reduction
quantifies the practical gain from selective prediction in a deployment context.

A non-monotone curve suggests uncertainty is not well-calibrated at the
sample level, even if global Spearman is high (depth-index confound).