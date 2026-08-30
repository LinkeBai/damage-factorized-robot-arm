# ICRA 16-day primary-platform reset

Date: 2026-08-30

## Decision

The original five-positioning-joint arm is restored as the primary platform.
`sim/assets/arm_push.xml` uses the same J1--J5 chain and limits as
`sim/assets/arm.xml`; it adds the planar block task and contact geometry.  It is
appropriate for primary kinematic/task simulation, while unmeasured inertia,
friction, backlash, gripper geometry, and camera calibration remain sim-to-real
limitations.

GenkiArm and Panda are transfer embodiments only.  Existing results on them are
retained, including No-Go outcomes, but cannot substitute for original-arm
prediction, action-ranking, closed-loop, and ablation evidence.

## Forty-eight-hour critical path

1. Validate original-arm task reachability and contact under intact, D2, and D3.
2. Run the five matched methods and three decisive ablations on seeds 7/17/27.
3. Stop methods that fail zero locked violation or fail to improve action
   ranking and endpoint error in at least two seeds.
4. Freeze the best supported method before physical evaluation; do not tune on
   physical evaluation episodes.
5. Execute the physical minimum: intact/D2/D3, fixed targets, repeated Push,
   synchronized joint state, two eye-to-hand videos, cube trajectory, abort and
   safety ledger.

## Sixteen-day evidence package

- Original-arm Push main table with five confirmation seeds.
- Constraint, prediction, action-ranking, closed-loop, and compute metrics.
- Projection/path-support/paired-effect attribution ablations.
- Held-out lock, lock angle, target, and physics perturbation tests.
- GenkiArm/Panda frozen-method transfer table, explicitly secondary.
- Grasp short-lift feasibility; method comparison only if the task baseline is
  repeatable before the method is tested.
- Physical original-arm repeated-trial table and videos.
- Every table generated from machine-readable results; failed runs remain in the
  ledger.

## Current evidence interpretation

The current paper is not yet a 4/5 ICRA paper.  The reset prevents a category
error in which transfer-platform evidence was treated as primary evidence.  A
4/5 candidate requires a meaningful novelty delta, matched baselines, decisive
mechanism ablations, and original-arm control-relevant gains.  Work volume alone
does not satisfy this gate.
