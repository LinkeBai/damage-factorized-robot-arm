# BT-DPWM reaction stabilization: Z6--Z8

## Frozen target

Against shared h136/240 (338,102 parameters), require three-seed/four-domain mean
free, object and overall improvements above zero, mean overall at least 5%, at
most one regressing cell, and exact damage projection. BT-DPWM remains fixed.

## Results

- Z6 validation selection (epoch zero included): selected epochs 40/0/40 for
  seeds 7/17/27. Final mean free -0.71%, object +22.00%, overall +0.24%, with
  6/12 regressions. Validation did not predict seed27 test degradation.
- Adapter selectivity diagnosis: contact/non-contact output-norm ratios were
  1.18, 0.76 and 0.99. The learned correction was not contact selective.
- Analytic geometry diagnosis: contact gap median was about -11.5 mm and its
  90th percentile about -8.0 mm; non-contact 10th percentile was about -1.5 mm.
- Z7 parameter-free current-contact gate: parameter count remains 338,056.
  Retrained seed7 reached free +2.34%, object +25.57%, overall +3.73%, 0/4
  regressions. A frozen-adapter gate chosen on seed7 (threshold +5 mm,
  temperature 2 mm) reached three-seed means free +0.77%, object +21.99%,
  overall +1.68%, with 5/12 regressions.
- Z8 one-step gated residual identification: seed7 free +0.80%, object +25.54%,
  overall +2.27%, with 1/4 regressions.

All Z6--Z8 gates are NO-GO. The independent object expert remains consistently
strong. The unresolved failure is robot reaction stability over long rollouts:
a deployable correction needs geometry-triggered event memory with bounded
decay, not a memoryless gate or unconstrained recurrent correction.
