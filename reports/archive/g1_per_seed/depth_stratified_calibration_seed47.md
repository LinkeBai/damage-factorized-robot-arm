# Depth-Stratified Calibration Analysis

**Seed**: 47  **Domain**: D3__mixed_composition

## Key Finding

- Global Spearman (all depths merged): **+0.9731**
- Depth-stratified mean Spearman: **+0.7981**

## Interpretation

The depth-stratified Spearman is computed per depth-step, then averaged.
Within each step, both uncertainty and error are concentrated in a narrow
range — the cross-depth variance (which drives the global correlation) is
removed. The remaining within-step variance may be much lower, producing a
low per-depth correlation even when the overall calibration is strong.

## Per-Depth Statistics

| Depth | N | Mean U | Mean E | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 45 | 0.0432 | 0.0528 | +0.8201 |
| 1 | 45 | 0.1019 | 0.1356 | +0.8692 |
| 2 | 45 | 0.1707 | 0.2370 | +0.7867 |
| 3 | 45 | 0.2434 | 0.3422 | +0.7738 |
| 4 | 45 | 0.3163 | 0.4432 | +0.8006 |
| 5 | 45 | 0.3883 | 0.5370 | +0.8198 |
| 6 | 45 | 0.4591 | 0.6254 | +0.8253 |
| 7 | 45 | 0.5289 | 0.7084 | +0.7858 |
| 8 | 45 | 0.5979 | 0.7836 | +0.7505 |
| 9 | 45 | 0.6663 | 0.8528 | +0.7488 |

## Conclusion

If within-step Spearman is low but mean_U and mean_E both grow with depth,
this confirms the global correlation is depth-index driven (a confound),
not evidence of calibrated uncertainty. The ensemble disagreement tracks
rollout horizon but not individual prediction difficulty within a step.