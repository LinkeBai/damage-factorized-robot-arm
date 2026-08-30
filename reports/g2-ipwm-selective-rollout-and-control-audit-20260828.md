# Selective IPWM Prediction and Closed-Loop Audit (2026-08-28)

## Decision

- **Prediction and state-isolation mechanism: PASS (retrospective three-seed audit).**
- **Closed-loop control-improvement claim: NO-GO.**
- **50 mm Push success metric: INVALID for the current target split.** The
  initial block-to-goal distance is only 30--40 mm, so a zero-contact episode
  can be labelled successful before acting.

## Failure that motivated the repair

The original full-state IPWM intervention was not safe to describe as a
uniform improvement.  On seed 7, D3 mixed-unseen, H50, it increased free-joint
RMSE by 6.01% relative to the shared projected model.  A paired trajectory
audit found a 95% interval of [0.06885, 0.07538] for the MSE increase and 93.3%
of terminal windows were worse.  This is a systematic coupling failure, not an
outlier.

## Structural repair

`SelectiveInterventionRollout` maintains two internal trajectories:

1. a mechanism-matched no-intervention carrier for robot/free coordinates;
2. the full coupled IPWM trajectory for object coordinates.

The published rollout concatenates carrier robot coordinates with IPWM object
coordinates and applies analytic topology projection.  Consequently, robot,
pusher and locked-coordinate predictions are exactly equal to the carrier
baseline, while the intervention retains its own internal robot--object
feedback.  Unit tests include a robot-coupled toy model to verify this
separation over multiple rollout steps.

## Frozen open-loop result

The physical-support threshold (context norm >= 1.2) was frozen after seed 27.
Across seeds 27/37/47, the three routed domains (D3 high damping, mixed
composition and mixed unseen), and horizons 10/25/50:

- object RMSE improved in 27/27 cells;
- cell improvements ranged from 1.34% to 39.97%;
- mean seed-level improvement was 17.04%;
- the three-seed bootstrap 95% interval was [6.92%, 26.34%];
- free-joint and pusher changes were exactly 0%;
- locked-coordinate violation RMSE was exactly zero.

The three-seed interval is descriptive, not definitive.  The isolation wrapper
was designed after inspecting seeds 37 and 47, so a genuinely untouched seed is
still required before treating this as confirmatory evidence.

Machine-readable result:
`runs/g2_ipwm_selective_rollout_20260828/summary.json`.

## Closed-loop No-Go

A frozen guarded-CEM smoke first reproduced the metric defect: with 30 approach
and 40 push steps, all methods had zero contact and zero displacement but were
marked successful by the 50 mm tolerance.  The audit therefore uses 60 approach
and 90 push steps and reports terminal distance, contact, and displacement
rather than the invalid success flag.

For D3 high damping over three held-out targets, mean terminal distances were:

| Seed | Nominal IK | Carrier MPC | Selective IPWM MPC | IPWM vs carrier |
|---:|---:|---:|---:|---:|
| 27 | 16.99 mm | 13.75 mm | 19.45 mm | -41.5% |
| 37 | 16.99 mm | 18.77 mm | 15.67 mm | +16.5% |
| 47 | 16.99 mm | 34.25 mm | 39.11 mm | -14.2% |

The direction is inconsistent, so the current paper must not claim that IPWM
improves closed-loop control.  The defensible claim is narrower: selective
intervention improves object prediction in routed physical-OOD regimes without
changing the carrier's robot/free-state prediction.

## Publication consequences

1. Present the seed-7 free-joint regression as the motivating failure and the
   state-isolation wrapper as the technical repair.
2. Keep the complete physics-spectrum table, including exact fallback ties.
3. Do not report 50 mm task success for this target split.
4. State that prediction-to-control transfer remains unresolved.
5. Remove all real-robot placeholders and make the submission explicitly
   simulation-only.
