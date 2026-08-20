# FTC-WM Gate L Audit

**Date:** 2026-08-21
**Seed:** 7
**Frozen budget:** 60 epochs, 150-step trajectories

## Decision

**NO-GO.** The explicit contact/free-object branch converged without numerical
divergence but did not enter the matched baseline loss range and did not reduce
object rollout error from the K2 scale.

## Optimization

| Epoch | FTC-WM loss | Matched baseline loss |
|---:|---:|---:|
| 1 | 0.3990 | 0.0831 |
| 20 | 0.1705 | 0.0243 |
| 30 | 0.1149 | 0.0215 |
| 40 | 0.0759 | 0.0205 |
| 60 | 0.0371 | 0.0176 |

The final FTC-WM loss remained approximately `2.11x` the matched baseline.
The gradient norm decreased throughout training, so this is a fidelity failure
under the frozen budget rather than numerical divergence.

## Rollout Evidence

| Domain | Matched object RMSE | FTC-WM object RMSE |
|---|---:|---:|
| D3 mixed composition | 0.0249 | 0.2456 |
| D2 mixed composition | 0.0371 | 0.2615 |
| D4 mixed composition | 0.0228 | 0.2209 |
| D3 mixed unseen | 0.0339 | 0.2596 |

Mean FTC-WM object RMSE was approximately `0.247`, versus approximately
`0.103` for FT-GWM K2 v2. Gate L therefore worsened the K2 object scale by
about `2.4x`. The run summary reports `18.15%` free-arm regression,
`885.63%` object regression, and `gate_passed=false`.

## Scope

This result closes the post-K2 contact/free-object architecture attempt. Do not
add capacity, contact features, loss reweighting, or extra epochs without a new
mechanism and a newly frozen protocol. The supported G2 direction remains
ordinary ensemble averaging plus selective prediction; FT-GWM K1 is retained
only as a provisional constraint-preserving joint-dynamics result.

## Artifacts

- `config/experiment/g2_ftcwm_gate_l_v1.yaml`
- `runs/g2_ftcwm_gate_l/seed7_v1/results.csv`
- `runs/g2_ftcwm_gate_l/seed7_v1/summary.json`
