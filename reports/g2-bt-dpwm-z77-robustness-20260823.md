# BT-DPWM Z77 Robustness Matrix

## Frozen matrix

Before results were observed, commit `fc82072` froze six residual conditions
(backlash, two-step delay, payload/armature, motor loss, damping/friction, and a
held-out composition), topologies D2/D3/D4, seeds 57/67, and nested budgets
0/5/10/25/50. Models, adapters, encoder, Z75 gate, calibration excitation, and
evaluation targets were unchanged. This yields 180 paired rollout rows.

## Aggregate result

| transitions | 0 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| BT own gain (%) | 0.000 | 2.105 | 2.234 | 5.316 | 5.276 |

Every seed/domain/budget BT-own gain is non-negative and every locked-coordinate
violation is zero. K50 mean own gain is 9.04% for seed57 and 1.51% for seed67.
Thus the robustness safety gate passes.

Strict budget monotonicity does not pass. The aggregate curve falls by 0.041
percentage points from K25 to K50. Seed67 D2 payload falls from 8.19% own gain
at K25 to 6.20% at K50 after a replacement context, and D2 friction falls from
7.88% at K10 to 6.66% at K25. Neither becomes worse than its K0 model, but both
contradict a strong claim that every additional transition is non-degrading.

## Factor-stratified K50 result

| factor | mean BT-own gain (%) | seed57 | seed67 |
|---|---:|---:|---:|
| backlash | 6.555 | 12.172 | 0.937 |
| delay | 2.464 | 3.383 | 1.545 |
| payload | 8.048 | 14.029 | 2.066 |
| motor | 6.785 | 11.273 | 2.296 |
| friction | 7.802 | 13.385 | 2.220 |
| held-out composition | 0.000 | 0.000 | 0.000 |

The held-out composition result is conservative rather than adaptive: Z75
retains z=0 at every budget for both seeds and all three topologies. It therefore
shows safe fallback, not useful adaptation to the hardest composition.

## Failure diagnosis

The D2 friction K25 context barely clears hysteresis: mean nested-support loss
improves 2.038% against a required 2%, while independent task rollout loses
1.217 percentage points of own gain. D2 payload K50 is more informative: all
four stored support windows improve and the mean support score improves 6.421%,
yet independent rollout loses 1.995 points versus its K25 incumbent. Constraint
projection remains exact, so the failure is not geometric. It is a calibration-
to-task generalization gap from a single active-probing trajectory.

The result supports the narrow claim that Z75 prevents catastrophic robustness
regression below K0. It does not support strict transition-by-transition
monotonicity under all residual shifts, and it does not establish adaptation on
the hardest held-out composition. Improving this within BT-DPWM requires a
development-only risk estimate that better predicts independent goal-rollout
change; retuning hysteresis on seeds 57/67 is prohibited.

Authoritative artifact:

- `runs/g2_bt_dpwm_z77_robustness/two_seed_summary_v1/summary.json`
