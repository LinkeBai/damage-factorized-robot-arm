# G2 Z82--Z85 BT-DPWM structural ablations (2026-08-23)

## Protocol

The four ablations were frozen in commit `6230156` before evaluation and use the
unseen Z76 confirmation seeds 57/67, identical domains and nested budgets. Each
row is paired against the corresponding frozen Z76 row. These are mechanism
diagnostics, not new threshold-development runs.

| Ablation | Selected comparison | Max paired overall change | Max paired object change | Max violation RMSE |
|---|---:|---:|---:|---:|
| Z82: remove analytic projection | all 40 rows | 0.00656 | 2.04e-5 | 0.14454 |
| Z83: allow locked-coordinate residual | all 40 rows | 0 | 0 | 0 |
| Z84: add residual after object transition | K>0, 32 rows | 4.43e-6 | 1.97e-5 | 0 |
| Z85: force nonzero context at K=0 | K=0, 8 rows | 0.05399 | 7.36e-5 | 0 |

Values above are absolute paired RMSE differences. The machine-readable source
is `runs/g2_bt_dpwm_z82_structural_ablations/two_seed_summary_v1/summary.json`.

## Findings

1. **Analytic projection is necessary for the hard safety claim.** Removing it
   produces locked-coordinate violation RMSE up to 0.14454. Seed67 also loses
   roughly 1--2.5% relative performance across the printed domain rows. The
   zero-violation result of the full model is therefore not an incidental neural
   prediction property.
2. **The adapter's internal locked-coordinate mask is defense in depth.** When
   only this mask is removed, the deployed predictions remain exactly identical
   because the final analytic projection removes those components. It should be
   retained for interpretable intermediate states, but it is not an independent
   source of final-output accuracy or safety under the current pipeline.
3. **The robot-to-object block-triangular bridge is exercised but its measured
   marginal effect is small.** Moving the residual after the object transition
   changes adapted object RMSE by at most 1.97e-5. This confirms a causal code
   path, but does not justify claiming that the bridge alone explains the main
   performance gain.
4. **Exact K=0 bypass is behaviorally material.** Replacing the zero context by
   one fixed norm-matched vector changes overall RMSE by as much as 0.05399 and
   causes the adapter proposal to be used at K=0. This ablation is a sensitivity
   test, not evidence that this particular arbitrary direction is generally
   harmful; its role is to show why the deployed model must enforce exact base
   recovery rather than merely hope a learned context is close to zero.

## Claim boundary

The structural evidence supports the complete safety construction: exact K=0
base recovery, reversible support-gated adaptation, and final analytic topology
projection. It does **not** repair the failed Z76 paired-equivalence gate, the
Z77 strict budget-monotonicity failure, or the Z79 raw-spread rollout-risk
calibration failure. Therefore G2 remains evidence-complete only after a unified
synthesis, and its current scientific verdict remains a narrow safe-adaptation
claim rather than performance superiority or calibrated rollout-risk.
