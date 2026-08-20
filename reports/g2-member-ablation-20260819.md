# G2 Member Count Ablation Report

**Date**: 20260819
**Ablation**: ensemble member count 1 / 3 / 5
**Method**: ordinary deep ensemble (constant condition mode)
**Seeds**: 5  **Bootstrap CI**: 95%

## Results

### D2 (seen topology)

| Members | Mean RMSE | 95% CI | vs M=1 |
|---:|---:|:---:|---:|
| 1 | 0.7137 | [+0.6145, +0.8619] | +0.0% |
| 3 | 0.5009 | [+0.4496, +0.5674] | +29.8% |
| 5 | 0.5015 | [+0.4494, +0.5824] | +29.7% |

### D3 (seen topology in G2 original, held-out in heldout experiment)

| Members | Mean RMSE | 95% CI | vs M=1 |
|---:|---:|:---:|---:|
| 1 | 0.6783 | [+0.5761, +0.8315] | +0.0% |
| 3 | 0.4507 | [+0.3914, +0.5146] | +33.5% |
| 5 | 0.4466 | [+0.3913, +0.5263] | +34.2% |

## Key Findings

1. **D2**: M=3 reduces RMSE by 29.8% vs M=1; M=5 reduces by 29.7% vs M=1.
   Marginal gain from M=3 to M=5: -0.1%.
2. **D3**: M=3 reduces RMSE by 33.5% vs M=1; M=5 reduces by 34.2% vs M=1.
   Marginal gain from M=3 to M=5: 0.9%.

## Conclusion

Increasing member count from 1 to 3 yields substantial RMSE reduction.
The marginal gain from 3 to 5 is smaller, supporting M=3 as the default choice
used in the main G2 experiments (good accuracy/compute tradeoff).