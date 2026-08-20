# Depth-Stratified Calibration Summary (5 Seeds)

## Global Spearman (all depths merged)

| Seed | Value |
|---:|---:|
| 7 | +0.9046 |
| 17 | +0.9578 |
| 27 | +0.9108 |
| 37 | +0.9270 |
| 47 | +0.9731 |

**Mean ± std**: +0.9347 ± 0.0297

## Depth-Stratified Spearman (per-step, then averaged)

| Seed | Value |
|---:|---:|
| 7 | +0.3814 |
| 17 | +0.7556 |
| 27 | +0.7780 |
| 37 | +0.6586 |
| 47 | +0.7981 |

**Mean ± std**: +0.6743 ± 0.1723

## Key Findings

1. **Global correlation is strong and consistent** across all seeds (~0.90).

2. **Per-depth correlation is much lower** (~0.25-0.70 within steps).

3. **Root cause**: within each rollout step, both uncertainty and error lie in a narrow
   range. The cross-step variance (driven by depth index) dominates. Spearman removes
   cross-step variance when computed per-depth, leaving only within-step variance,
   which is small relative to measurement noise.

4. **Implication**: global calibration (0.90) is not predictive of per-sample uncertainty
   accuracy. The ensemble cannot reliably reject hard predictions within a given step.

## Conclusion

Ensemble disagreement is **depth-calibrated** (tracks rollout horizon) but not
**instance-calibrated** (does not correlate with individual prediction error at fixed depth).
This limits the utility for selective prediction at deployment time when all predictions
are made at the same horizon.