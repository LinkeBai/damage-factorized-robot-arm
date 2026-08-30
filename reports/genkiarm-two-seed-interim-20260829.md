# Calibrated GenkiArm two-seed interim result

This report is an interim audit after completing preregistered training seeds
107 and 117. It is not a replacement for the required five-seed analysis.

## Primary effects versus the mechanism-matched carrier

| Method | Seed 107 | Seed 117 | Two-seed mean | Positive fraction |
|---|---:|---:|---:|---:|
| Routed selective IPWM | +0.6037% | -0.7836% | -0.0899% | 1/2 |
| Raw selective IPWM | +1.4198% | -1.9133% | -0.2467% | 1/2 |

For both methods and both seeds, free-joint regression is exactly 0% and the
maximum locked-coordinate violation RMS is 0. The selective state-isolation
invariant therefore replicates, but the object-prediction benefit does not.

The two-seed hierarchical bootstrap intervals cross zero:

- routed: [-0.9643%, +0.7749%];
- raw selective: [-2.2758%, +1.7089%].

Training seed is the population-level unit, so these intervals remain interim.

## Frozen-gate implication

The routed primary method currently fails the +5% mean-effect threshold, the
0.8 positive-seed fraction threshold, and the positive lower confidence bound.
With the current sum of effects, seeds 127/137/147 would need to average about
+8.39% routed improvement to bring the final five-seed mean to +5%. This is
mathematically possible but inconsistent with the first two observed effects.

The appropriate current conclusion is: **replicated state-isolation mechanism,
No-Go for performance superiority**. No threshold or seed list is changed.

Authoritative artifact:
`runs/g2_ipwm_genkiarm_confirmation_v2/two_seed_interim_summary.json`.
