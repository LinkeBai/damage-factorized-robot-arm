# Shared IPWM cross-arm robot-transition Gate

**Decision:** SMALL GATE PASS (2/3 SEEDS); PROMOTE ONLY TO OBJECT/CONTACT GATE

## Frozen protocol

One shared variable-node model was jointly trained on all five GenkiArm joints
and all seven Panda joints.  Per robot, intact plus two non-middle single-lock
conditions were training data; GenkiArm j3 and Panda joint4 were held out, as
were their test episodes.  Inputs contain full joint state, normalized action,
lock mask/angle and local MJCF axis/origin, but no robot identity, solver force,
future contact or deleted joints.

The shared graph has 42,626 parameters.  The flat padded MLP has 41,614
parameters (2.37% difference).  Fairness review corrected both models to use
the same `analytic projection + learned state delta` parameterization before
the final run; preliminary absolute-state MLP results were discarded.

## Held-out-lock results

RMSE is over full valid q/qvel nodes in independent held-out episodes.

| Seed | Flat Genki | Shared Genki | Flat Panda | Shared Panda | Pooled improvement | Seed decision |
|---:|---:|---:|---:|---:|---:|---|
| 7 | 0.02786 | 0.01923 | 0.09203 | 0.02631 | 67.45% | PASS |
| 17 | 0.01576 | 0.02164 | 0.07439 | 0.02430 | 59.76% | FAIL: Genki regression |
| 27 | 0.01835 | 0.01402 | 0.03499 | 0.02791 | 20.77% | PASS |

All three pooled improvements exceed 10%, but the frozen rule additionally
requires both robots to improve.  Seed 17 fails that requirement, leaving 2/3
positive seeds and a narrow Gate pass.  The shared model also beats the
analytic inertial baseline on both robots in all seeds.

## Claim boundary

This demonstrates that a single shared structural mechanism can learn
free-joint transitions for two different full kinematic chains and generalize
to an unseen locked joint within each chain.  Both robot structures appear in
training, so it is **not** zero-shot transfer to an unseen robot.  It contains
no object/contact prediction, Grasp, visual observation, action ranking or
closed-loop evidence.  The seed-17 GenkiArm regression is a required failure
boundary.  The result permits only the next object/contact small Gate and does
not yet satisfy the senior's full cross-robot/cross-task evidence request.

Artifacts: `runs/ipwm_cross_arm_prediction_gate_v1/seed{7,17,27}.json` and
`summary.json`.
