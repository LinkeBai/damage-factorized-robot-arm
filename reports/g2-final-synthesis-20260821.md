# Final G2 Evidence Synthesis

**Date:** 2026-08-21
**Seeds:** 7, 17, 27, 37, 47
**Protocol SHA-256:** `bfb43ff804f91b538efdcd7c8a22bf5812de901b344d8603afad88eddfe73055`

## Decision

The supported main result is ordinary ensemble averaging plus selective
prediction. The evidence does not support a distinct topology-conditioning or
complete structured-world-model advantage. FT-GWM K1 remains a narrower
constraint-preserving joint-dynamics contribution.

## Prediction Evidence

| Comparison | Mean improvement | Seed bootstrap 95% CI | Positive seeds | Decision |
|---|---:|---:|---:|---|
| G1 corrected 3-member ensemble vs parameter-matched single | 30.74% | [15.06%, 42.62%] | 5/5 | Supported |
| G2 structured vs ordinary ensemble | 2.47% | [-1.83%, 6.38%] | 4/5 | CI crosses zero |
| Held-out D3 topology conditioning | 0.02% | [-3.38%, 3.70%] | 2/5 | CI crosses zero |

The D3 member ablation improves mean RMSE from
`0.6783` (one member) to
`0.4507` (three members), a
`33.55%` reduction.

## Selective Prediction

| Coverage | Mean RMSE | Mean reduction | Across-seed std |
|---:|---:|---:|---:|
| 70% | 0.3102 | 30.96% | 2.93% |
| 50% | 0.2218 | 50.50% | 2.98% |

All five coverage-RMSE curves are monotone. However, global uncertainty-error
Spearman (`0.935 +/- 0.030`)
drops to `0.674 +/- 0.172`
after stratifying by rollout depth. Therefore disagreement is a useful
rejection score on the evaluated mixed-depth rollout distribution, but it must
not be described as fully instance-calibrated at a fixed horizon.

This synthesis corrects an indexing error in the older selective-prediction
summary: the actual reductions are `30.96%` at 70% coverage and `50.50%` at
50% coverage.

## Compute

| Method | Parameters | Mean train seconds | Std seconds | Device |
|---|---:|---:|---:|---|
| Structured ensemble | 450,906 | 97.0 | 14.5 | CUDA |
| Ordinary deep ensemble | 450,906 | 87.6 | 3.8 | CUDA |

Both G2 ensembles use three members, 20 epochs, 150-step trajectories and the
same parameter count. Wall-clock values are measured end-to-end training times
from the five frozen run summaries; GPU model was not recorded, so no
cross-machine compute claim is permitted.

## Structural Branch

- **FT-GWM K0:** PASS for exact fixed-SE(3) kinematics.
- **FT-GWM K1:** two-seed provisional PASS for zero violation and free-joint fidelity.
- **FT-GWM K2:** NO-GO for complete Push prediction; object RMSE regressed 986.08%.
- **FTC-WM Gate L:** NO-GO. The isolated contact/free-object branch converged
  without divergence, but finished at loss `0.0371` versus the matched baseline
  `0.0176`. Mean object rollout RMSE was approximately `0.247`, about `2.4x`
  worse than K2 v2 (`0.103`), with `885.63%` object regression.

## Frozen Claims

Supported:

- A three-member ordinary ensemble materially improves prediction over a single model.
- Ensemble disagreement supports selective rejection on the evaluated rollout mixture.
- FT-GWM K1 exactly satisfies known joint locks with provisional joint fidelity.

Not supported:

- Topology conditioning outperforms an ordinary deep ensemble.
- Disagreement is fully instance-calibrated at fixed rollout depth.
- FT-GWM is a complete Push object/contact model.
- Prediction improvements already imply statistically established control gains.

## Next Decision

Do not reopen DFWM/CR-GWM/RC-GWM/FT-GWM/FTC-WM head tuning. Before G3, freeze the
paper tables and decide whether the narrower ensemble/selective-prediction
claim justifies real-robot evaluation. Any G3 uncertainty gate must be
calibrated at the deployment horizon, not from pooled rollout depths.
