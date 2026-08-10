# G1 Pivot Control Baseline

Date: 2026-08-10

Status: **COMPLETE - deterministic baseline passes; learned MPC remains NO-GO**

The Jacobian-transpose damped Reach controller verifies that the calibrated
MuJoCo environment is controllable independently of the learned world model.
Evaluation uses the frozen four-target evaluation split, a 50 mm tolerance,
and at most 1,000 MuJoCo steps.

| Morphology | Success | Final distances (m) |
|---|---:|---|
| intact | 4/4 | 0.048, 0.046, 0.048, 0.046 |
| D2 | 4/4 | 0.050, 0.048, 0.049, 0.049 |
| D3 | 3/4 | 0.048, 0.048, 0.109, 0.049 |

Global-sampling IK initialization plus joint PD removes the remaining local
minimum. The resulting reference controller reaches 4/4 evaluation targets
for intact, D2, and D3. A controller-induced trajectory collector adds bounded
exploration around this baseline for task-relevant world-model training.

## Learned MPC Recheck

The world model was changed to residual-state prediction and trained with a
five-step autonomous rollout loss. Controller-induced data improved the
one-seed prediction smoke by about 13% and produced a positive D3 control
smoke. A formal frozen-MPC recheck then used D2/D3, seeds 7/17/27, K=0/1/2/5,
one frozen evaluation target per damage, 64 CEM candidates, and 400 steps.

- Seed 7: DFWM K=1 succeeds on D2 and D3; topology-only fails both.
- Seed 17: no method succeeds; DFWM final distance improves over topology-only.
- Seed 27: no method succeeds; DFWM is worse than topology-only.

Only 1/3 seeds shows the required control success, so the 2/3-seed G1 gate
still fails. The controller proves the environment and target split are
controllable; the remaining problem is learned-model/planner seed stability.
Artifacts: `results/final/g1-control-pivot-20260810/`.
