# Post-freeze D3 confirmation protocol (registered before generation)

Registration time: 2026-08-31 (Asia/Shanghai)

## Scope and honesty boundary

This is a one-pass, post-freeze confirmation of the already trained seed 7, 17,
and 27 checkpoints on a newly generated candidate archive. D3 has appeared in
earlier exploratory work, so this is **not** claimed as a pristine unseen-domain
test. The untouched unit is the candidate/query archive generated after this
registration. No model, loss, checkpoint, planning budget, success threshold,
or reporting metric may be changed after reading its results.

## Frozen choices

- Robot/model: `sim/assets/arm_push.xml`, the original 5-DoF simulation model.
- Fault: D3 joint lock.
- Candidate RNG seed: `91031`.
- Data size: 40 episodes, 5 replans per episode, 128 candidates per replan.
- Horizon: 5 action segments x 10 simulator steps = 50 steps.
- Checkpoints: seeds 7, 17, and 27 already frozen under
  `runs/icra_primary_decision_full_w10/` and
  `runs/icra_primary_global_matched_w10/`.
- Comparisons: nominal vs same-capacity global residual; selective IPWM vs
  same-capacity global residual; analytic projection vs no projection.
- Primary metrics: top-1 regret, terminal candidate error, success rate,
  Spearman action rank, contact-response RMSE, and lock violation.
- All three model seeds and all generated groups are retained.

## Predeclared interpretation

- **Strong transferable advantage:** regret reduction is positive for all 3
  seeds and its aggregate relative reduction is at least 20%.
- **Moderate transferable advantage:** positive for all 3 seeds and aggregate
  reduction is 10--20%.
- **Weak/inconclusive:** fewer than 3 positive seeds or aggregate reduction
  below 10%.
- Selective structural attribution passes only if selective IPWM beats the
  same-capacity global residual in at least 2/3 seeds on regret and terminal
  error, without worsening the aggregate of either metric.
- Response RMSE, contact rate, terminal error, and success are reported even
  when they contradict the regret result. No subgroup will replace the full
  result as the primary claim.

## Run policy

The archive is generated once. Each frozen checkpoint is evaluated once on the
same archive. A rerun is allowed only for a documented crash or implementation
bug that is covered by a regression test; the failed output remains in the
provenance ledger.
