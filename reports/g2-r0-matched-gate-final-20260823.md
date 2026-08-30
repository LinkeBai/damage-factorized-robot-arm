# G2-R0 Matched-Adapter Frozen Gate

## Frozen method

The evaluated model is the contact-aware block-coordinate/intervention-projected
BT-DPWM family.  It retains analytic lock projection, a frozen Z69 robot block,
a frozen shared compact object head, a support-aware latent intervention
residual, and explicit pusher-geometry propagation.  An existing observable Z65
8D physical-context posterior modulates the intervention residual through a
bounded centered gate `h(z)-h(0)`.  K=0 is therefore an exact bypass after any
gate training.  The frozen K25 policy uses posterior scale 1.38, eight zero-gate
rollout steps, and depth ramp 0.06.

The strongest baseline is shared h136 + analytic damage projection + its
compute/data/architecture-matched Z70 rank-8 adapter.  BT uses the corresponding
same-protocol Z70 rank-8 adapter and the same Z65 K25 support trajectory.  The
adapter is applied before the BT object block so object prediction remains a
downstream consequence of the adapted projected robot transition.

## Frozen three-seed D3 result

Object RMSE improvement over the matched shared baseline at H10/H25/H50:

| Seed | Mixed composition | Mixed unseen |
|---:|---:|---:|
| 7 | +12.27 / +5.33 / +37.84% | +15.89 / +32.50 / +36.60% |
| 17 | +6.53 / +13.54 / +12.52% | +9.10 / +15.50 / +2.95% |
| 27 (untouched confirmation) | +7.62 / +2.04 / +18.98% | +9.19 / +2.85 / +2.58% |

All 18 cells are positive and exceed the frozen 2% object gate.  The minimum
confirmation value is 2.0430%.  Constraint violation is exactly zero.  On D3,
free/overall do not materially regress; the observed pusher differences remain
small in absolute task-space units.

## K efficiency and long-horizon mechanism

On seed17, K0 H50 is -122.55/-96.62% against the matched baseline.  K25 with
context but without depth risk leaves mixed-unseen H50 at -3.86%.  The complete
K25 method changes H50 to +12.52/+2.95%.  Across the earlier frozen projected
audit, K5/K10/K50 already turn the seed17 H50 catastrophe into positive values,
but the curve is not monotonic; K25 is the selected deployment budget.  We do
not claim monotonic improvement with K.

## Controls and ablations

Intact/D2/D4 and D2/D4 mixed controls show at most 1.72% absolute object
difference across all seeds/horizons under matched adapters.  Free-joint changes
remain within the 5% non-regression band.  The largest relative IID pusher
regression is amplified by a small denominator; its absolute difference is
0.798 mm, and the maximum checked IID absolute pusher difference is 1.081 mm.

Seed17 matched component ablations show:

- removing both intervention branches leaves only about -2.46% to +1.35%;
- removing geometry reduces composition H10 from +6.53% to +0.08% and
  mixed-unseen H50 from +2.95% to +1.17%;
- removing the latent branch reduces mixed-unseen H10/H25 from +9.10/+15.50%
  to +3.54/+2.50%;
- removing depth risk makes mixed-unseen H50 regress by 3.86%.

Thus analytic intervention routing, explicit robot-to-object geometry, latent
correction, observable context, and rollout-risk scheduling each have distinct
evidence.  The machine-readable source of truth is
`runs/g2_r0_physical_context_residual/matched_gate_summary_v2.json`.

## Claim boundary

Because the robot block consumes contact/object context, the implementation is
not a strict forward block-triangular dynamical graph.  The defensible claim is
a contact-aware block-coordinate, intervention-projected world model with a
stop-gradient robot-to-object bridge and explicit geometric propagation.  These
results close G2-R simulation evidence; they do not replace real-arm validation.
