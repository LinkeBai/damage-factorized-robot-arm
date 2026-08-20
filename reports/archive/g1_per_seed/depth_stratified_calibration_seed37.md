# Depth-Stratified Calibration Analysis

**Seed**: 37  **Domain**: D3__mixed_composition

## Key Finding

- Global Spearman (all depths merged): **+0.9270**
- Depth-stratified mean Spearman: **+0.6586**

## Interpretation

The depth-stratified Spearman is computed per depth-step, then averaged.
Within each step, both uncertainty and error are concentrated in a narrow
range — the cross-depth variance (which drives the global correlation) is
removed. The remaining within-step variance may be much lower, producing a
low per-depth correlation even when the overall calibration is strong.

## Per-Depth Statistics

| Depth | N | Mean U | Mean E | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 45 | 0.0396 | 0.0455 | +0.7246 |
| 1 | 45 | 0.0963 | 0.1090 | +0.6602 |
| 2 | 45 | 0.1644 | 0.1846 | +0.5735 |
| 3 | 45 | 0.2400 | 0.2628 | +0.5820 |
| 4 | 45 | 0.3209 | 0.3395 | +0.6212 |
| 5 | 45 | 0.4060 | 0.4117 | +0.6386 |
| 6 | 45 | 0.4943 | 0.4801 | +0.6501 |
| 7 | 45 | 0.5853 | 0.5453 | +0.7091 |
| 8 | 45 | 0.6786 | 0.6059 | +0.7099 |
| 9 | 45 | 0.7738 | 0.6631 | +0.7163 |

## Conclusion

If within-step Spearman is low but mean_U and mean_E both grow with depth,
this confirms the global correlation is depth-index driven (a confound),
not evidence of calibrated uncertainty. The ensemble disagreement tracks
rollout horizon but not individual prediction difficulty within a step.