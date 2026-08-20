# Selective Prediction Experiment

**Seed**: 7  **Domain**: D3__mixed_composition

## Setup

Ensemble disagreement (epistemic uncertainty = member std) is used as a
rejection score. For each coverage fraction, the lowest-uncertainty samples
are retained and RMSE is computed on that subset.

## Key Metrics

- Global uncertainty-error Spearman: **+0.9046**
- Coverage-RMSE Spearman (monotonicity check): **+1.0000**
- Is monotone (> 0.9): **Yes**
- Baseline RMSE (100% coverage): **0.3497**

## Selective Prediction Curve

| Coverage | N Retained | RMSE | RMSE Reduction |
|---:|---:|---:|---:|
| 100% | 450 | 0.3497 | +0.00% |
| 90% | 405 | 0.3222 | +7.85% |
| 80% | 360 | 0.2926 | +16.32% |
| 70% | 315 | 0.2591 | +25.90% |
| 60% | 270 | 0.2258 | +35.43% |
| 50% | 225 | 0.1915 | +45.24% |
| 40% | 180 | 0.1556 | +55.50% |
| 30% | 135 | 0.1166 | +66.67% |
| 20% | 90 | 0.0824 | +76.43% |

## Interpretation

If RMSE drops monotonically as coverage decreases, the ensemble uncertainty
successfully identifies hard predictions. At 50% coverage the RMSE reduction
quantifies the practical gain from selective prediction in a deployment context.

A non-monotone curve suggests uncertainty is not well-calibrated at the
sample level, even if global Spearman is high (depth-index confound).