# Z65 Uncertainty-Calibrated BT-DPWM — Three-Seed Gate

## Result

The deployment-adaptation mechanism passes the frozen seed 7/17/27 gate.
All BT-own gains are measured against the same seed/domain model at K=0.

| seed | mean gain at K=0/5/10/25/50 (%) | positive domains at K=50 |
|---:|---|---:|
| 7 | 0 / 3.45 / 4.78 / 5.77 / 5.77 | 4/4 |
| 17 | 0 / 0 / 0 / 7.79 / 10.98 | 3/4 |
| 27 | 0 / 0 / 0 / 2.03 / 7.70 | 3/4 |

Across every seed, domain, and budget there is no negative BT-own transfer.
Every seed-level mean curve is monotonic. Maximum locked-coordinate violation
is zero. Maximum absolute object-RMSE change is 0.173%; the model contains no
object-side residual bypass, so this change is mediated by calibrated robot
transitions.

## Mechanism retained

- known-topology analytic projection;
- K=0 exact zero residual context;
- free-joint-only residual innovation;
- robot-to-independent-object block-triangular rollout;
- physical-context posterior inferred from state/action/known mask only;
- nested-budget consistency and per-axis uncertainty;
- validation rollback, posterior precision fusion, topology observability wait,
  and replacement hysteresis.

## Remaining risk

The adaptation mechanism is stable, but the frozen K=0 BT base has substantial
seed variance against shared h136/240. Seed 17 is weaker overall; seed 27 has a
stronger free arm and overall score but a weaker object expert. The next gate is
therefore base checkpoint/training stability followed by a fair shared comparison,
not another replacement architecture.

Authoritative artifact:
`runs/g2_bt_dpwm_context_encoder_z65/three_seed_gate_v1/summary.json`.
