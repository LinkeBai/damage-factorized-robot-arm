# Depth-Stratified Calibration Analysis

**Seed**: 7  **Domain**: D3__mixed_composition

## Key Finding

- Global Spearman (all depths merged): **+0.9046**
- Depth-stratified mean Spearman: **+0.3814**

## Interpretation

The depth-stratified Spearman is computed per depth-step, then averaged.
Within each step, both uncertainty and error are concentrated in a narrow
range — the cross-depth variance (which drives the global correlation) is
removed. The remaining within-step variance may be much lower, producing a
low per-depth correlation even when the overall calibration is strong.

## Per-Depth Statistics

| Depth | N | Mean U | Mean E | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 45 | 0.0543 | 0.0404 | +0.6970 |
| 1 | 45 | 0.1193 | 0.0930 | +0.5030 |
| 2 | 45 | 0.1919 | 0.1537 | +0.4376 |
| 3 | 45 | 0.2702 | 0.2174 | +0.4185 |
| 4 | 45 | 0.3526 | 0.2782 | +0.3384 |
| 5 | 45 | 0.4381 | 0.3340 | +0.2424 |
| 6 | 45 | 0.5259 | 0.3866 | +0.2934 |
| 7 | 45 | 0.6154 | 0.4360 | +0.3267 |
| 8 | 45 | 0.7064 | 0.4823 | +0.2904 |
| 9 | 45 | 0.7987 | 0.5268 | +0.2665 |

## Conclusion

If within-step Spearman is low but mean_U and mean_E both grow with depth,
this confirms the global correlation is depth-index driven (a confound),
not evidence of calibrated uncertainty. The ensemble disagreement tracks
rollout horizon but not individual prediction difficulty within a step.