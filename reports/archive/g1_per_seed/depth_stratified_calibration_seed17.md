# Depth-Stratified Calibration Analysis

**Seed**: 17  **Domain**: D3__mixed_composition

## Key Finding

- Global Spearman (all depths merged): **+0.9578**
- Depth-stratified mean Spearman: **+0.7556**

## Interpretation

The depth-stratified Spearman is computed per depth-step, then averaged.
Within each step, both uncertainty and error are concentrated in a narrow
range — the cross-depth variance (which drives the global correlation) is
removed. The remaining within-step variance may be much lower, producing a
low per-depth correlation even when the overall calibration is strong.

## Per-Depth Statistics

| Depth | N | Mean U | Mean E | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 45 | 0.0408 | 0.0454 | +0.8332 |
| 1 | 45 | 0.0875 | 0.1021 | +0.7994 |
| 2 | 45 | 0.1364 | 0.1689 | +0.7815 |
| 3 | 45 | 0.1871 | 0.2416 | +0.7507 |
| 4 | 45 | 0.2394 | 0.3158 | +0.7093 |
| 5 | 45 | 0.2931 | 0.3881 | +0.6911 |
| 6 | 45 | 0.3478 | 0.4582 | +0.6876 |
| 7 | 45 | 0.4035 | 0.5265 | +0.7167 |
| 8 | 45 | 0.4600 | 0.5918 | +0.7937 |
| 9 | 45 | 0.5171 | 0.6555 | +0.7927 |

## Conclusion

If within-step Spearman is low but mean_U and mean_E both grow with depth,
this confirms the global correlation is depth-index driven (a confound),
not evidence of calibrated uncertainty. The ensemble disagreement tracks
rollout horizon but not individual prediction difficulty within a step.