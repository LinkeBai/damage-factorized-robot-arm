# IPWM solver-native constraint-response Gate status

**Date:** 2026-08-28  
**Decision:** **NO-GO FOR DEPLOYABLE PROPAGATION OPERATOR**. Implementation,
pre-contact, and exact-prefix solver-label identifiability pass; held-out-joint
deployable estimation fails, so action-ranking/control promotion is forbidden.

## Why this is not a renamed CR-GWM

The prior CR-GWM injected the nominal model's predicted lock-position and
lock-velocity residual into a learned graph message adapter. Its five-seed raw
metrics were positive, but a parameter-matched model and an unconstrained
adapter prevented causal attribution. Subsequent reaction variants were also
unstable under held-out physics.

The only new hypothesis allowed here is narrower: a solver-native joint-lock
constraint exposes a low-dimensional reaction force that may explain the
free-joint counterfactual delta. If that explanatory relation is weak, this
route stops and CR-GWM is not reopened.

## Completed implementation check

- Added inactive MuJoCo equality constraints for `j1`--`j5` at model load.
- A selected equality can be activated at an arbitrary lock angle per episode.
- Under a sustained command opposing a D3 lock, the locked generalized
  constraint force is non-zero.
- After 100 steps the solver-native lock position error is below `2e-5 rad`.
- Other inactive lock equalities do not constrain their joints.
- The deployed analytic projection remains responsible for the stricter
  `1e-7` feasibility requirement.
- Repository regression suite: **259 passed**.

## Next required evidence

1. Generate paired intact/locked rollouts from identical initial states and
   actions using both the legacy pinning environment and the solver-native
   constraint environment.
2. Verify that changing the lock simulator does not manufacture task-level
   gains; report trajectory differences explicitly.
3. Regress free-joint counterfactual state deltas from the measured lock
   reaction, split before and after tool--object contact.
4. Require at least 70% explained variance before implementing a learned
   response head.
5. If the threshold passes, compare against prior CR-GWM and an unstructured
   parameter-matched residual under the frozen three-seed protocol.

No paper score or novelty claim is increased by the current implementation
check alone.

## Pre-contact paired result

Using 768 same-state/same-action pairs across D2/D3/D4-style locks (`j2`,
`j3`, `j4`), with the block moved away to exclude contact:

| Metric | Result |
|---|---:|
| Raw held-out-sample explained variance | 0.99547 |
| Train-only scalar calibration | 0.93698 |
| Calibrated held-out-sample explained variance | 0.99999 |
| Calibrated held-out RMSE | 0.000101 rad/s |
| Actual counterfactual delta RMS | 0.03708 rad/s |
| Per-lock calibrated R2 | j2 0.999998; j3 0.999992; j4 0.999947 |

This passes the preregistered 0.70 explanatory threshold. It demonstrates that
the solver-native equality force is the correct low-dimensional cause of the
one-step free-joint counterfactual delta in the no-contact regime. It is also
close to a direct consequence of constrained dynamics, so it is **not** by
itself a learning contribution or novelty result.

Artifact: `runs/ipwm_constraint_response_gate_v1/precontact_seed20260828.json`.

The route may continue only to the harder checks: separation from contact
impulse, equivalence to the legacy pinning task distribution, inference from
deployable observations, held-out-joint generalization, action ranking, and
closed-loop control.

## Exact-prefix contact-phase result on the calibrated GenkiArm model

The earlier attempt to branch from v4's saved 14-D observation was rejected:
the saved trigger flag described contact in the preceding integration step,
while the reconstructed current geometry had no contact.  A 14-D observation
also cannot generally reconstruct solver warm-start, delay, or backlash state.

The replacement diagnostic rolls the calibrated GenkiArm Push model to real
tool--block contact, copies the complete `MjData`, and only then activates a
solver-native lock in one branch at the current angle.  The intact and locked
branches have maximum prefix-qpos difference exactly `0`; both receive the
same next action.  Train/test splits are by episode, not by transition.

| Seed | Paired rows | free joints: equality only R2 | free joints: equality + contact-delta R2 | object: equality only R2 | object: equality + contact-delta R2 |
|---:|---:|---:|---:|---:|---:|
| 7 | 267 | 0.98359 | 0.99990 | -0.34066 | 1.00000 |
| 17 | 234 | 0.97142 | 0.99991 | -0.19202 | 1.00000 |
| 27 | 255 | 0.96235 | 0.99988 | -0.33001 | 1.00000 |

Artifacts: `contact_prefix_seed{7,17,27}.json/.npz` under
`runs/ipwm_constraint_response_gate_v1/`.

Interpretation is deliberately narrow.  The result proves that the local lock
reaction explains most free-joint response, while object response requires the
change in contact constraints: a contact-mediated propagation path is present
and is directionally stable across all three seeds.  The near-perfect combined
score is a constrained-dynamics decomposition using solver labels, not a
deployable predictor and not a learned-method result.  It therefore does not
pass the overall Gate and does not raise the paper score.  Promotion now
requires estimating the propagation term from deployable state/action/fault
inputs, holding out `j3`, and beating a parameter-matched unstructured model in
action ranking and closed-loop control.

## Deployable held-out-joint Gate: No-Go

The final preregistered learning check used only deployment-available inputs:
14-D state, 5-D candidate action, and the continuous lock location plus its
local state/action.  No equality force, contact force, future state, or solver
label was supplied as an input.  Training used only j2/j4 locks; j3 and the
test episodes were both held out.  The shared path model had 6,337 parameters
versus 6,151 for the unstructured MLP (3.02% difference).

| Seed | Test rows | unstructured RMSE | structured RMSE | structured relative change | structured R2 |
|---:|---:|---:|---:|---:|---:|
| 7 | 29 | 1.0906 | 1.3393 | -22.80% | -8.5710 |
| 17 | 27 | 0.9281 | 0.5927 | +36.14% | -0.7854 |
| 27 | 22 | 0.7321 | 1.3748 | -87.79% | -6.2214 |

Only 1/3 seeds is directionally positive; all structured held-out R2 values
are negative.  This fails before action-ranking and closed-loop evaluation.
Per the frozen rule, the candidate is stopped: no hidden-width search, extra
attention/gating module, seed selection, or threshold revision is allowed.

The scientific conclusion is useful but negative: the solver-native causal
decomposition is real and stable, yet the current state/action representation
does not identify a transferable j2/j4-to-j3 propagation law.  Consequently
CFPO-style language cannot be promoted to the paper's core innovation, and
these solver results may appear only as mechanism diagnosis/failure boundary.

Artifacts: `deployable_propagation_seed{7,17,27}.json` under
`runs/ipwm_constraint_response_gate_v1/`.
