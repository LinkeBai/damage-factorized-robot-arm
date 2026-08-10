# G1 Control Diagnosis

## Evidence

The frozen learned-MPC gate used one evaluation target (`eval_00`), one episode
per condition, horizon 5, 32 CEM candidates, and two CEM iterations. Most runs
failed, with only one successful D3 episode across the three seeds.

The same target and damage domains were then tested with the deterministic
global-sampling IK plus joint PD controller:

| Domain | IK residual (m) | Reach result | Steps | Final error (m) |
|---|---:|---|---:|---:|
| D2 mixed composition | 0.0312 | pass | 74 | 0.0499 |
| D3 mixed composition | 0.0003 | pass | 82 | 0.0485 |

This rules out the primary hypotheses that `eval_00` is unreachable, that the
MuJoCo damage pinning is broken, or that the FK target convention is invalid.

## Diagnosis

The current failure is in the learned control chain:

1. A short-horizon CEM planner is asked to control a delayed, damaged system
   with only 32 candidates and two iterations.
2. The world model has one-step prediction evidence, but its rolled-out state
   predictions are not yet accurate enough for 300-step receding-horizon use.
3. The control gate evaluates only one target and one episode per condition,
   so it is a useful failure probe but not a stable success-rate estimate.
4. Because topology-only also fails, the result does not isolate factorization
   as the cause; it primarily shows that the learned-MPC deployment is not yet
   reliable.

## Next diagnostic

Keep the environment and target fixed and sweep only the planner:

- horizon: 5, 10, 15;
- candidates: 32, 128, 256;
- iterations: 2, 4, 6;
- compare learned-MPC against IK+PD on D2 and D3.

If planner scaling does not recover the target, the next test is a world-model
multi-step rollout error check. The G1 gate remains No-Go until that control
chain is repaired or the project pivots to conditional dynamics plus a proven
controller.
