# G1 active-calibration diagnostic (2026-08-18)

## Problem reproduced

The first active-probe pilot used deployment-only square-wave probes with
`lr=0.1`, `l2=0.001`, and a latent range of `[-5, 5]`. The inferred latent norm
reached about 5.75 and multi-step RMSE increased from 0.1918 at K=0 to 0.3034
at K=5. This was calibration-distribution mismatch plus latent overfitting.

## Repair

- mixed two active-probe and two random trajectories per training domain;
- reduced latent learning rate to 0.01;
- increased L2 regularization to 0.1;
- clipped each latent coordinate to `[-1, 1]` and its gradient norm to 1;
- reserved a sixth probe trajectory for validation-selected early stopping;
- retained K trajectories only for optimization and kept evaluation disjoint;
- added exact `tool_geom`--`block_geom` contact detection and an
  identifiability diagnostic.

All 115 tests pass after the change.

## Seed-7 gate result

| K | DFWM multi-step RMSE | Mean latent norm |
|---|---:|---:|
| 0 | 0.238282 | 0.000 |
| 1 | 0.235454 | 0.530 |
| 2 | 0.235293 | 0.524 |
| 5 | 0.235236 | 0.515 |

The repair prevents destructive adaptation and restores a monotonic K trend.
K=5 improves over K=0 by only 1.28%, below the predeclared 5% seed-7 mechanism
threshold. The 3-seed expansion was therefore not run.

## Decision

The optimization failure is fixed, but the few-shot residual signal remains
too weak for the paper's main claim. Do not resume the five-seed benchmark yet.
The next scientific choice is either to add explicit residual supervision /
contrastive identification during training, or pivot to the zero-shot result.

The active probes separate residual profiles in joint-state space but produced
no tool-block contact in the current scene. They are valid joint-dynamics
diagnostic sequences, not evidence of a goal-directed Push task.
