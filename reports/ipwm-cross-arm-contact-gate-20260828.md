# Shared IPWM cross-arm contact/object Gate

**Decision:** NO-GO (0/3 SEEDS)

## Data integrity

The frozen generator produced 80 current-contact prefixes per robot and three
lock interventions per prefix: 240 GenkiArm and 240 Panda counterfactual rows.
No prefix was rejected, intact/locked branch qpos/qvel differences before the
intervention were exactly zero, and both branches retained post-step contact
in 98.75% of GenkiArm and 100% of Panda rows.  Object-response RMS was nonzero
and similar across robots (`0.13066` and `0.12936`), so failure is not caused by
an all-zero target.  Splits group all three locks from the same prefix together
to prevent state leakage.

Training uses j2/j4 for GenkiArm and joint2/joint6 for Panda; held-out tests use
j3 and joint4 respectively.  Both models see the same deployment-observable
full joint state/action/lock geometry, object pose/twist and end-effector--object
relative position.  Robot identity, solver force, future contact and future
state are forbidden.

The shared graph/object model has 66,731 parameters and the flat MLP 66,889
(0.24% difference).  Both predict the same nine-dimensional object position +
twist counterfactual delta and use identical grouped splits and target scaling.

## Result

| Seed | Flat pooled RMSE | Structured pooled RMSE | Relative improvement | Per-robot boundary |
|---:|---:|---:|---:|---|
| 7 | 0.10567 | 0.13525 | -28.00% | both regress |
| 17 | 0.10970 | 0.17945 | -63.58% | both regress |
| 27 | 0.10955 | 0.12399 | -13.18% | Genki improves slightly; Panda regresses |

The preregistered requirement was at least 10% pooled improvement with both
robots improving in at least 2/3 seeds.  The observed result is 0/3 positive
seeds.  Seed 7's Panda structured prediction is even slightly worse than the
zero-response baseline; seed 17 is substantially worse on GenkiArm.

## Consequence

The earlier robot-transition Gate remains a narrow positive result: shared
chain structure helps free-joint prediction for held-out locks in 2/3 seeds.
It does **not** propagate reliably to object/contact response.  Consequently:

- no cross-arm object-propagation or full-IPWM generalization claim is allowed;
- no Push/Grasp five-seed expansion is justified for this candidate;
- no object-head width, loss weighting, extra contact feature, gating or seed
  search is permitted after reading this result;
- solver-native decomposition may remain only as a diagnostic/failure boundary.

Artifacts: `runs/ipwm_cross_arm_contact_gate_v1/dataset_*` and
`seed{7,17,27}.json`.
