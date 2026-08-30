# G2 Dual-Expert World Model — Gate Q0-B

**Date:** 2026-08-21

**Decision:** NO-GO (3/5 seeds; frozen requirement 4/5)

**Config:** `config/experiment/g2_dual_expert_gate_q0b_v1.yaml`

## Question

Does cross-expert joint discrepancy add risk information beyond ordinary object
ensemble disagreement at fixed rollout depth, and does an untrained equal-weight
rank combination improve selective AURC by at least 10%?

The evaluation uses the held-out `D3__mixed_composition` domain. Scores are
rank-normalized separately at every rollout depth, preventing rollout depth from
creating a spurious uncertainty-error correlation. No score weights are fitted.

## Frozen gate

- Mean fixed-depth selective AURC improvement: at least 10% per passing seed.
- Mean fixed-depth partial Spearman of `u_cross` and realized fused-model error,
  controlling object ensemble disagreement: positive.
- At least 4/5 seeds must pass.

## Results

| Seed | AURC improvement | Partial Spearman | Seed decision |
|---:|---:|---:|---|
| 7 | +18.93% | +0.583 | PASS |
| 17 | +11.23% | +0.573 | PASS |
| 27 | +6.46% | +0.522 | NO-GO |
| 37 | +15.01% | +0.551 | PASS |
| 47 | +0.57% | +0.104 | NO-GO |
| Mean | +10.44% | +0.467 | 3/5 |

All five partial correlations are positive, so structural discrepancy contains
some conditional error information in this protocol. However, the preregistered
operational claim fails: the equal-weight combined risk score exceeds the 10%
AURC threshold in only 3/5 seeds. The mean improvement cannot override the
seed-consistency requirement.

## Decision and boundary

Q0-B is **NO-GO**. Do not proceed to Q0-C or Guarded MPC under the current
DE-DWM core-method claim. Do not tune score weights, thresholds, or coverage
ranges after observing these seeds.

Q0-A remains a supported engineering result: product-space fusion preserves
object prediction, improves free-arm prediction in the evaluated seeds, and
maintains exact constraints. It is not sufficient as the paper's central risk
mechanism. The positive partial correlations may be reported only as exploratory
evidence or used to motivate a future independently preregistered method, not as
a passed result.
