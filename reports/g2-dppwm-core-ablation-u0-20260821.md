# DPP-WM Core Ablation — Gate U0

**Date:** 2026-08-21

**Decision:** NO-GO for the claim that internal projection is required for the
prediction gain

## Question

Does the complete combination of independent joint/object transitions,
step-wise analytic projection, and recurrent fused-state feedback provide a
prediction advantage that disappears when any component is removed?

## Frozen seed-7 gate

At depth 10, complete DPP-WM must improve free-arm RMSE by at least 10% over:

- monolithic ensemble with projection inside every rollout step;
- independent product experts without projection;
- independent product experts with projection only on reported output, not on
  the recurrent feedback state.

Object regression must be at most 2%, and constraint violation must be zero.

## Results

| Method | Depth-1 free | Depth-5 free | Depth-10 free | Depth-10 object | Violation |
|---|---:|---:|---:|---:|---:|
| monolithic autonomous | 0.07 | 0.48 | 0.90 | 0.50 | 1.38 |
| monolithic internal projection | 0.07 | 0.48 | 0.93 | 0.49 | 0 |
| product no projection | 0.04 | 0.20 | 0.44 | 0.48 | 0.30 |
| product output-only projection | 0.04 | 0.20 | 0.44 | 0.48 | 0 |
| complete DPP internal projection | 0.04 | 0.20 | 0.44 | 0.48 | 0 |

Complete DPP improves `52.86%` over monolithic internal projection, but only
`0.14%` over product no projection and `0.14%` over product output-only
projection. The preregistered gate is therefore NO-GO.

## Interpretation

The strong predictive gain comes from training and rolling out separate joint
and object transition models. Moving analytic projection into the recurrent
feedback loop does not materially improve free-joint or object prediction in
this seed. Projection remains valuable because it changes constraint violation
from `0.30` to exactly zero, but it is a safety/feasibility guarantee rather than
the source of the 53% prediction improvement.

The previously proposed core statement—independent transitions, internal
projection, and fused feedback are jointly necessary for stability—is not
supported. The surviving method is a factorized joint/object world model with
an analytic damage projection guarantee. Its empirical value is strong, but its
novelty relative to standard modular/factorized dynamics remains unresolved.

Do not expand U0 across seeds under the failed internal-projection claim.
