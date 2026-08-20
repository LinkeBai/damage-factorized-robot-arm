# Selective Prediction Experiment

**Seed**: 37  **Domain**: D3__mixed_composition

## Setup

Ensemble disagreement (epistemic uncertainty = member std) is used as a
rejection score. For each coverage fraction, the lowest-uncertainty samples
are retained and RMSE is computed on that subset.

## Key Metrics

- Global uncertainty-error Spearman: **+0.9270**
- Coverage-RMSE Spearman (monotonicity check): **+1.0000**
- Is monotone (> 0.9): **Yes**
- Baseline RMSE (100% coverage): **0.4402**

## Selective Prediction Curve

| Coverage | N Retained | RMSE | RMSE Reduction |
|---:|---:|---:|---:|
| 100% | 450 | 0.4402 | +0.00% |
| 90% | 405 | 0.3746 | +14.88% |
| 80% | 360 | 0.3401 | +22.74% |
| 70% | 315 | 0.2954 | +32.90% |
| 60% | 270 | 0.2529 | +42.55% |
| 50% | 225 | 0.2118 | +51.87% |
| 40% | 180 | 0.1680 | +61.83% |
| 30% | 135 | 0.1283 | +70.86% |
| 20% | 90 | 0.1002 | +77.23% |

## Interpretation

If RMSE drops monotonically as coverage decreases, the ensemble uncertainty
successfully identifies hard predictions. At 50% coverage the RMSE reduction
quantifies the practical gain from selective prediction in a deployment context.

A non-monotone curve suggests uncertainty is not well-calibrated at the
sample level, even if global Spearman is high (depth-index confound).