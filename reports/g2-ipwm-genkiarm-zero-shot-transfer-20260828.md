# Frozen IPWM zero-shot transfer to calibrated GenkiArm Push

**Date:** 2026-08-28  
**Decision:** PARTIAL PREDICTION TRANSFER; NOT A DEPLOYMENT OR CONTROL GO

## Protocol

The seed 27/37/47 IPWM checkpoints, matched few-shot adapters, physical-context
encoders, threshold `1.2`, target split, 150-step trajectories and H10/H25/H50
evaluation were frozen.  No component was retrained or calibrated on the new
model.  The only change was replacing the hard-coded simplified
`arm_push.xml` simulator with `genkiarm_push.xml`, whose calibrated kinematics,
CAD appearance and collision proxies are documented separately.  Independent
query seeds 127/137/147 generated new trajectories; the absolute XML path is
part of each cache key.

This is an exploratory audit because the actual-model transfer protocol was
created after the original checkpoints.  It cannot be called confirmatory.

## Results

| Seed | Context norm / route | H10 selective | H25 selective | H50 selective | Routed result |
|---:|---|---:|---:|---:|---|
| 27 | 1.170 / carrier | +15.43% | +2.52% | -10.83% | three ties |
| 37 | 1.115 / carrier | +20.04% | +25.19% | -3.61% | three ties |
| 47 | 1.270 / IPWM | +11.61% | +23.85% | +13.18% | same three gains |

Percentages are object-RMSE improvement relative to the mechanism-matched
carrier.  Raw selective IPWM improves 7/9 cells and has seed-mean improvement
`10.82%`, with a three-seed bootstrap interval `[2.38%, 16.21%]`.  However,
the frozen deployment router activates IPWM for only seed 47: routed behavior
improves 3/9 cells, ties 6/9, and has interval `[0.00%, 16.21%]`.

Safety remains exact by construction: routed free-joint RMSE change is zero
in every cell and locked-coordinate violation RMSE is zero.  This supports the
claim that selective projection/isolation transfers safely to the calibrated
model, but not that deployment reliably benefits from it.  H50 regressions in
seeds 27/37 also show that raw intervention is not uniformly robust.

## Publication consequence

The result may be used as exploratory cross-model evidence for safe prediction
isolation.  It cannot satisfy the requested second-robot criterion because the
kinematic structure is still the same GenkiArm, cannot establish cross-task
generalization, and cannot repair the existing closed-loop No-Go.  The current
paper therefore remains below the frozen 4.0/5 bar.

Machine-readable summary:
`runs/g2_ipwm_genkiarm_zero_shot_v1/summary.json`.
