# GenkiArm evidence ledger and V2 confirmation freeze

## Decision

The existing five-seed, six-method Push matrix is **not** evidence on the
calibrated GenkiArm model. Its manifest identifies
`sim/assets/arm_push.xml`. It may remain as simplified-model development
evidence, but it must not be described as validation on the actual 5-DoF arm
model.

The only completed calibrated-GenkiArm SI-IPWM result is the frozen zero-shot
prediction transfer in `runs/g2_ipwm_genkiarm_zero_shot_v1`. It evaluates
checkpoint seeds 27/37/47 on `sim/assets/genkiarm_push.xml`. Raw selective
IPWM improves object prediction in 7/9 domain-horizon cells, but the frozen
router activates only for seed 47. This is three-seed exploratory prediction
evidence, not a five-seed confirmation and not positive closed-loop evidence.

## Evidence ledger

| Artifact | Robot model | Seeds | What it can support | What it cannot support |
|---|---|---:|---|---|
| `runs/g1_push_6methods_5seeds/20260818-formal` | simplified `arm_push.xml` | 5 | development comparison on a simplified arm | actual GenkiArm, external validity, hardware |
| `runs/g2_ipwm_selective_rollout_20260828` | simplified `arm_push.xml` | 27/37/47 | retrospective SI-IPWM mechanism diagnosis | clean confirmation; actual GenkiArm |
| `runs/g2_ipwm_selective_rollout_query57_20260828` | simplified `arm_push.xml` | checkpoint seeds 27/37/47 | held-out trajectory replication | a fourth training seed |
| `runs/g2_ipwm_genkiarm_zero_shot_v1` | calibrated `genkiarm_push.xml` | 27/37/47 | frozen zero-shot open-loop prediction transfer | five-seed confirmation; closed-loop superiority |
| operational/action-ranking diagnostics | calibrated or audited task interfaces as stated per report | 7/17/27 or 27/37/47 | failure-boundary evidence | a positive control claim |

## Seed-integrity correction

The V1 evidence contract reserved seeds 57/67/77/87/97. Later work used 57
and 67 for BT-DPWM confirmation and robustness, and used 77/87 in context
calibration. They are therefore no longer untouched for a newly frozen
SI-IPWM confirmation. Reusing them would create a post-selection ambiguity.

Before any V2 training or evaluation, the replacement seed set is frozen as
`107/117/127/137/147` in
`config/experiment/icra_2027_genkiarm_confirmation_v2.yaml`. A literal audit
of `runs/`, `config/`, `reports/`, and `scripts/` found no existing `seed107`,
`seed117`, `seed127`, `seed137`, or `seed147` artifacts at freeze time.

## What must be rerun

All five V2 seeds require fresh training with the frozen SI-IPWM recipe and
the calibrated GenkiArm simulator. Evaluation must use disjoint deterministic
trajectory seeds, the same method list, the same horizons, at least three lock
locations, four physics families, and ten trajectories per condition.

The evaluation order is fixed:

1. analytic feasibility, projection idempotence, and state-isolation tests;
2. five-seed open-loop prediction matrix on calibrated GenkiArm;
3. failure-boundary diagnostics for action ranking and closed-loop Push;
4. Grasp feasibility/scripted-control baseline;
5. Panda simulation-only external-validity check and dual eye-to-hand camera
   perturbation study.

Closed-loop Push remains diagnostic until it passes the predeclared gate. A
failed gate narrows the paper to prediction, hard feasibility, selective state
isolation, and guarded fallback. Failure is not positive evidence and must not
be rhetorically converted into “realism.”

## Immediate execution consequence

The current checkpoint inventory contains the dependencies needed for seeds
27/37/47 and later contaminated seeds, but not for all five newly frozen V2
seeds. Therefore a valid five-seed GenkiArm matrix cannot be obtained by only
rerunning the evaluator; it requires fresh checkpoint training first.
