# Selective Prediction Summary (5 Seeds)

## Baseline Metrics

- **Uncertainty-error Spearman** (global, merged): +0.9347
- **Baseline RMSE** (100% coverage): 0.4507
- **Monotonicity** (coverage-RMSE Spearman > 0.95 for all seeds): ✓ Yes

## Selective Prediction Curve (Mean ± Std across 5 seeds)

| Coverage | N Samples | Mean RMSE | Std RMSE | Mean Reduction | Std Reduction |
|---:|---:|---:|---:|---:|---:|
| 100% | 2250 | 0.4507 | 0.0774 | +0.00% | 0.00% |
| 90% | 2025 | 0.3920 | 0.0611 | +12.78% | 2.83% |
| 80% | 1800 | 0.3536 | 0.0546 | +21.30% | 2.80% |
| 70% | 1575 | 0.3102 | 0.0485 | +30.96% | 2.93% |
| 60% | 1350 | 0.2664 | 0.0411 | +40.66% | 3.02% |
| 50% | 1125 | 0.2218 | 0.0318 | +50.50% | 2.98% |
| 40% | 900 | 0.1771 | 0.0274 | +60.50% | 3.01% |
| 30% | 675 | 0.1320 | 0.0203 | +70.55% | 2.75% |
| 20% | 450 | 0.0927 | 0.0154 | +79.26% | 2.83% |

## Key Results

1. **Perfect monotonicity across all seeds**: RMSE decreases monotonically as coverage increases,
   confirming uncertainty is a valid rejection score.

2. **At 50% coverage**: RMSE reduced by ~50.5% (±3.0%)
   with only half the predictions retained.

3. **Practical deployment**: An ensemble can trade off coverage for accuracy. At 70% coverage,
   ~31.0% error reduction is achievable.

## Interpretation

The perfectly monotone selective prediction curve demonstrates that ensemble disagreement
provides a **reliable uncertainty signal** for rejection. This is a strong positive result
for uncertainty-aware control and active learning applications.

## Conclusion

Ensemble disagreement is useful for **selective rejection on the evaluated mixed-depth
rollout distribution**. The depth-stratified audit prevents a stronger claim of full
instance-level calibration at a fixed deployment horizon.