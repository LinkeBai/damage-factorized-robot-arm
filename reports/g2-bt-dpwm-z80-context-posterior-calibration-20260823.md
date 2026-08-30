# BT-DPWM Z80 Physical-Context Posterior Calibration

## Protocol

Z80 evaluates Z65 as a posterior over the eight physical context coordinates,
not as a rollout-risk score. Five development encoder seeds use independent
active-probe trajectories to fit per-budget/per-dimension Gaussian variance
temperatures. Seeds 57/67 use separately seeded probes only for frozen coverage
evaluation. The NLL variance is correctly interpreted in normalized context
units defined by `CONTEXT_SCALE`.

The matrix contains 67 domains, two trajectories per domain, and budgets
3/6/15/30: 2,680 development and 1,072 confirmation records.

## Result

Mean dimensionwise absolute coverage error is 0.0856, below the preregistered
0.10 aggregate threshold. Temperature means for budgets 3/6/15/30 are
0.742/0.976/1.027/1.033; no value reaches the [0.05, 20] clipping bounds.

The aggregate pass hides important distribution-shape error:

| nominal coverage | mean dimensionwise absolute error |
|---:|---:|
| 50% | 0.226 |
| 80% | 0.056 |
| 90% | 0.030 |
| 95% | 0.030 |

The worst budget/dimension/coverage cell has absolute error 0.410. Errors are
similar by transition budget (0.083--0.089), so the weakness is central-interval
shape and coordinate heterogeneity rather than one budget alone.

## Decision

Gaussian temperature scaling passes the frozen aggregate MACE gate and supports
a narrow statement that high-coverage physical-context intervals are calibrated
on the evaluated distribution. It does not justify saying the full diagonal
Gaussian posterior is uniformly calibrated, especially around 50% coverage.
It also does not reverse Z79: physical-context coverage is distinct from
task-rollout risk ranking.

The same-posterior next step is per-budget/per-dimension conformal quantile
calibration. It requires new encoder seeds and independent probe draws because
the Z80 confirmation coverage has now been observed.

Authoritative artifact:

- `runs/g2_bt_dpwm_z80_context_posterior_calibration/summary.json`
