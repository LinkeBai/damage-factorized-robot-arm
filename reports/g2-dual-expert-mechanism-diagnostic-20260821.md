# Dual-Expert Post-Q0-B Mechanism Diagnostic

**Date:** 2026-08-21

**Status:** exploratory diagnosis; does not change the Q0-B NO-GO decision

## Question

Does cross-expert discrepancy predict local structural correction benefit, or
does product-space fusion help through a different mechanism?

Using the frozen five-seed Q0-A checkpoints and identical D3 evaluation
trajectories, the diagnostic compares `u_cross` with data-expert joint error,
structural correction gain, fused joint residual, object residual, and overall
residual at every fixed rollout depth.

## Results

Values below are mean fixed-depth partial Spearman correlations controlling
object ensemble disagreement.

| Seed | Data joint error | Correction gain | Fused joint residual | Object residual | Overall residual | Positive local correction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | +0.402 | -0.110 | +0.431 | +0.631 | +0.583 | 22.2% |
| 17 | +0.663 | +0.011 | +0.611 | -0.750 | +0.573 | 10.9% |
| 27 | +0.571 | -0.427 | +0.534 | +0.523 | +0.522 | 33.8% |
| 37 | +0.714 | -0.020 | +0.626 | -0.593 | +0.551 | 12.4% |
| 47 | -0.018 | -0.173 | -0.201 | +0.576 | +0.104 | 32.7% |

Mean local correction gain is negative in every seed (`-0.015` to `-0.040`
joint RMSE). Thus the structural expert is not generally a more accurate local
joint predictor when both experts receive the same fused state, and `u_cross`
does not measure local correction benefit.

## Mechanism interpretation

Q0-A compares autonomous ordinary-ensemble rollout with autonomous fused
rollout and reports large free-arm improvements. The local counterfactual above
shows that these improvements cannot be attributed to structural replacement
winning each individual step. The supported explanation is recurrent rollout
stabilization: exact locked-joint constraints and geometry-preserving joint
dynamics keep the state on the damaged-system manifold, preventing off-manifold
joint errors from feeding back into later object and joint predictions.

This also explains Q0-B. `u_cross` often measures current joint difficulty, but
after the structural transition has stabilized the state it is neither a stable
measure of remaining object error nor a stable measure of local intervention
gain. Combining it with object uncertainty therefore helps some seeds and harms
others.

## Consequence

Do not revive cross-expert residual risk as the core claim. A possible new
candidate is **constraint-manifold stabilized heterogeneous rollout**. It needs
a new preregistered attribution gate comparing:

1. ordinary autonomous ensemble;
2. ordinary ensemble plus direct analytic projection;
3. product fusion with FT joint dynamics;
4. product fusion with an unconstrained/matched joint expert;
5. teacher-forced or reset rollouts to separate local accuracy from feedback
   stability.

Only if FT product fusion suppresses depth-wise error growth beyond direct
projection and matched heterogeneous controls across seeds can stabilization be
claimed as a method contribution.
