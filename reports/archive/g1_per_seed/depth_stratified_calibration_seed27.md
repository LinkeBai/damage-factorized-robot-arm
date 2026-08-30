# Depth-Stratified Calibration Analysis

**Seed**: 27  **Domain**: D3__mixed_composition

## Key Finding

- Global Spearman (all depths merged): **+0.9108**
- Depth-stratified mean Spearman: **+0.7780**

## Interpretation

The depth-stratified Spearman is computed per depth-step, then averaged.
Within each step, both uncertainty and error are concentrated in a narrow
range — the cross-depth variance (which drives the global correlation) is
removed. The remaining within-step variance may be much lower, producing a
low per-depth correlation even when the overall calibration is strong.

## Per-Depth Statistics

| Depth | N | Mean U | Mean E | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 45 | 0.0390 | 0.0468 | +0.8157 |
| 1 | 45 | 0.0964 | 0.1155 | +0.7838 |
| 2 | 45 | 0.1654 | 0.1994 | +0.7971 |
| 3 | 45 | 0.2403 | 0.2888 | +0.7760 |
| 4 | 45 | 0.3182 | 0.3755 | +0.7691 |
| 5 | 45 | 0.3977 | 0.4561 | +0.7600 |
| 6 | 45 | 0.4782 | 0.5339 | +0.7770 |
| 7 | 45 | 0.5593 | 0.6087 | +0.7676 |
| 8 | 45 | 0.6407 | 0.6790 | +0.7690 |
| 9 | 45 | 0.7224 | 0.7456 | +0.7650 |

## Conclusion

If within-step Spearman is low but mean_U and mean_E both grow with depth,
this confirms the global correlation is depth-index driven (a confound),
not evidence of calibrated uncertainty. The ensemble disagreement tracks
rollout horizon but not individual prediction difficulty within a step.