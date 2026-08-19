# G1 Robust Zero-Shot Pivot (2026-08-19)

> Superseded for quantitative claims by
> `g1-robust-zero-shot-corrected-results-20260819.md`. The original Push
> collision/target protocol had zero contact and zero block displacement.

## Scope

The residual-identification claim is paused. The new mechanism candidate is a
topology-conditioned ensemble for robust zero-shot prediction, followed by
minimax Push MPC if the prediction gate remains stable.

## Seed-7 Medium Screen

Three independently initialized topology-only world models were trained on the
same Push data. On held-out D2 and D3 trajectories:

| Domain | Ensemble RMSE | Mean member RMSE | Relative gain | Depth-stratified disagreement/error Spearman |
|---|---:|---:|---:|---:|
| D2 | 0.4742 | 0.6224 | 23.8% | 0.565 |
| D3 | 0.4633 | 0.6120 | 24.3% | -0.170 |

The ensemble prediction gain is direction-consistent and substantial. Raw
disagreement/error correlation is confounded by rollout depth. After
depth-stratification, disagreement remains useful on D2 but is not calibrated on
D3. The models' predicted aleatoric variance is anti-calibrated and must not be
used as an MPC risk penalty.

## Decision

- **Go:** robust ensemble mean and minimax model-wise objective.
- **No-Go:** scalar uncertainty-penalty MPC using current log standard deviation
  or ensemble disagreement.
- **Next cheap gate:** compare ensemble-mean and minimax Push MPC on one held-out
  target in D2 and D3 using the saved medium checkpoint.
- **Promotion gate:** minimax must not reduce success in either domain and must
  improve worst-domain final block distance before multi-target or multi-seed
  runs.

Checkpoint:
`runs/g1_robust_zero_shot/seed7_medium_v1/ensemble.pt`

## Closed-Loop Screen

A staged controller used the same analytic-IK approach policy for all methods,
then compared nominal IK, ensemble-mean MPC, and minimax MPC. Short (`30+40`)
and protocol-length (`40+60`) screens produced zero block displacement in both
D2 and D3. One D2 ensemble-mean episode registered two contact steps, but still
did not move the block.

This is a benchmark/control-interface No-Go, not evidence against minimax MPC:
even the nominal controller did not produce a valid push. The control claim is
paused until a deterministic nominal Push episode moves the block on the exact
evaluation target. Prediction results remain valid because their evaluation
trajectories contain independently recorded contact/displacement coverage.
