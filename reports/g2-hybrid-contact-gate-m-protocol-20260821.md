# Gate M: Damage-Factorized Hybrid Contact Protocol

**Status:** M0/M1 complete; M2 proxy-impulse v1 NO-GO

## Hypothesis

The K2/FTC failure is caused by treating sparse contact and free motion as one
continuous object-state regression problem. A damage-compiled fixed-transform
chain plus an event-driven, unilateral, friction-constrained impulse operator
should improve contact transitions and multi-step object rollout.

## Stages

1. **M0 identifiability:** record a per-transition contact mask and verify at
   least 20 contact transitions in the frozen training set.
2. **M1 oracle reconstruction:** reconstruct target object velocity from the
   analytic free transition plus the target velocity impulse. RMSE must be at
   most `1e-6`; failure means the state slicing/integration protocol is wrong.
3. **M2 oracle-contact learned impulse:** train only the low-dimensional
   normal/tangential impulse with the true contact mask. It must improve object
   rollout RMSE by at least 30% over a parameter-matched continuous residual
   baseline under the same data and optimization budget.

## Stop Rules

- Fewer than 20 contact transitions: stop and repair data collection.
- M1 failure: stop and repair state/integration definitions.
- M2 below 30% improvement: stop the hybrid-contact method; do not implement a
  learned contact detector or run five seeds.
- M2 pass: freeze Gate M3 geometric contact detection before implementation.

## Scope

Gate M uses seed 7 only and does not authorize G3 real-robot statistics. The
existing ordinary ensemble remains the prediction baseline; FT-GWM K1 remains
the constraint-preserving joint backbone.

## 2026-08-21 Execution Note

- M0 passed with `916/3600` contact transitions.
- State-slice oracle velocity reconstruction passed at numerical zero.
- The initial proxy target, total object velocity change, is not a physical
  pusher impulse: `47.49%` of contact transitions have negative projected
  normal change and `71.18%` violate the nominal friction cone.
- Under the formal 60-epoch budget, the fair continuous operator reached
  rollout RMSE `0.133269`; the constrained proxy-impulse operator reached
  `0.337359`, a `153.14%` regression. M2 v1 is therefore **NO-GO**.
- Direct MuJoCo pusher/tool-to-block impulse extraction has now been added.
  Its aggregate vector still leaves object velocity-change RMSE `0.033050`,
  confirming that table friction and other contacts must be modeled separately.

No geometric contact detector or five-seed expansion is authorized. A future
M1b must use per-contact MuJoCo frames/impulses and a separate free/table
friction operator; it requires a newly frozen protocol rather than modifying
the failed M2 v1 result.

M1b was subsequently preregistered and executed. Its implicit momentum RMSE
was `0.002949` against a `0.001` threshold, and `29.32%` of exact aggregate
pusher impulses were negative in the single deployable contact basis. M1b
therefore also stopped before learned-model training; see
`reports/g2-hybrid-contact-gate-m1b-result-20260821.md`.
