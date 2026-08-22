# BT-DPWM Z81 Conformal Physical-Context Calibration

## Frozen protocol

Z81 keeps the Z65 encoder architecture and checkpoints unchanged. It calibrates
the standardized residual `|context error| / posterior std` with empirical
per-budget/per-dimension quantiles on development seeds 7/17/27/37/47. Encoder
seeds 77/87 and their independent active-probe trajectories were fixed before
any Z81 coverage result and are used only for confirmation.

The frozen gate requires overall, every nominal-coverage group, and every budget
group dimensionwise MACE to be at most 0.10.

## Result

| stratum | dimensionwise MACE |
|---|---:|
| overall | 0.0289 |
| nominal 50% | 0.0454 |
| nominal 80% | 0.0397 |
| nominal 90% | 0.0167 |
| nominal 95% | 0.0139 |
| budget 3 | 0.0326 |
| budget 6 | 0.0261 |
| budget 15 | 0.0260 |
| budget 30 | 0.0309 |

All frozen criteria pass. Compared with Z80 Gaussian temperature scaling, the
50% interval error falls from 0.226 to 0.045 without retraining or changing the
posterior mean. The calibration artifact contains the complete quantile radii
for four budgets, four coverages, and eight context dimensions.

## Claim boundary

This supports calling the physical-context posterior uncertainty calibrated on
the evaluated simulation distribution. It does not turn posterior width into a
task-rollout risk score: Z79 still shows that raw spread fails to rank harmful
adaptation proposals. The final mechanism therefore assigns distinct roles:

- conformal posterior intervals quantify physical-context estimation uncertainty;
- nested support validation and hysteresis accept or reject an update;
- permanent z=0 provides reversible deployment fallback;
- analytic projection enforces locked-coordinate safety exactly.

The obsolete absolute `mean_std <= 0.30` veto should not carry the safety claim.
Removing or replacing it in the executable method would be a method revision
requiring new end-to-end confirmation seeds; Z81 alone does not retroactively
change the Z76 decision.

Authoritative artifact:

- `runs/g2_bt_dpwm_z81_conformal_context_calibration/summary.json`
