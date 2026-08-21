# BT-DPWM X0--Y6 Mechanism Gates

Date: 2026-08-21. Primary domain: `D3__mixed_composition`. Seed: 7.
Baseline: frozen compute-matched shared topology graph (`h=96`, 120 epochs).

## Locked method

BT-DPWM is one world model with four executable properties:

1. a contact-conditioned robot graph block;
2. analytic damage projection after every transition;
3. an independent recurrent object graph block;
4. a directed gradient boundary plus block-coordinate, block-specific-horizon training.

The dual-expert work supplies the independent representations; BT-DPWM supplies
their directed coupling and exact damage constraint.  Gradient surgery is not a
separate method and is excluded from the accepted model.

## Causal sequence

| Gate | Single intervention | Free-arm | Object | Overall | Decision |
|---|---|---:|---:|---:|---|
| X0 | shared-trunk gradient audit | negative cosine in 47.5% of windows | joint/object norm ratio 43.3x | n/a | conflict confirmed |
| X1 | full-training object-preserving projection | -12.69% | +3.07% | -12.54% | NO-GO |
| Y0 | strict object-agnostic robot block | -8.92% | -40.45% | -9.28% | NO-GO |
| Y1 | restore current object/contact context | -3.72% | -38.40% | -4.13% | NO-GO |
| Y2 | independent recurrent object encoder | +0.86% | +2.55% | +0.87% | NO-GO (missed 1% by 0.13 pp) |
| Y3 | final 10-epoch joint-only refinement | -17.24% | +1.36% | -17.07% | NO-GO |
| Y4 | train both blocks at horizon 10 | +10.98% | +0.33% | +10.87% | NO-GO |
| Y5 | simultaneous block-specific horizons | +8.67% | -81.48% | +7.33% | NO-GO |
| **Y6** | **robot h10, freeze; object h5 on fixed robot rollouts** | **+4.16%** | **+3.69%** | **+4.16%** | **PASS** |

All percentages are improvements over the same frozen shared baseline at rollout
depth 10; positive is better.  Constraint violation RMS is zero for Y6.

## Mechanism interpretation

- Strict robot autonomy is false in pushing: current object/contact context is
  required for reaction dynamics.
- A joint-trained pooled representation is insufficient for object rollout; an
  independently trained object encoder is necessary.
- Joint-only terminal fine-tuning improves the training objective but destabilizes
  long rollout, so it is excluded.
- Robot and object blocks require different temporal credit horizons (10 and 5).
- Simultaneous optimization exposes the object block to a moving upstream robot
  rollout distribution.  Freezing the trained robot block before object training
  removes this non-stationarity and yields the first formal PASS.

## Current evidence boundary

Y6 is a seed-7 provisional mechanism pass, not yet a robustness claim and not yet
evidence that BT-DPWM exceeds DFWM across seeds or on hardware.  The architecture
and training schedule are now frozen.  Next work is replication, parameter/compute
accounting, DFWM comparison, and then real-robot validation if those gates pass.
