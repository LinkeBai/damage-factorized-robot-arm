# Stable BT-DPWM: final G2 evidence synthesis (2026-08-23)

## Decision

**G2 evidence delivery is complete; the broad G2 scientific gate is NO-GO.**
The evidence supports a narrower paper claim: a block-triangular, damage-projected
world model can perform reversible few-shot adaptation with exact K=0 recovery,
zero locked-coordinate violations, and nonnegative observed own-gain across the
frozen confirmation and robustness matrices. It does not establish superiority
to the compute-matched shared model, strictly monotone benefit with transition
budget, or rollout-risk calibration from raw posterior spread.

| Claim | Verdict | Frozen evidence |
|---|---|---|
| Safe nonnegative few-shot adaptation | PASS | Z76: all paired own-gains nonnegative; violation RMSE 0 |
| K25/K50 equivalence or superiority to shared | FAIL | Z76 K50 paired lower bound -1.191 pp, outside -1 pp margin |
| Robustness under six perturbation families | PARTIAL | Z77 no negative own-gain and zero violations; strict monotonicity failed |
| Support gate blocks observed harmful updates | PASS | Z79: 3 harmful proposals, all rejected; 0/11 accepted harmful |
| Raw posterior spread ranks rollout harm | FAIL | Z79 Spearman -0.734, wrong direction |
| Physical-context uncertainty coverage | PASS | Z81 conformal dimensionwise MACE 0.0289 on new encoder seeds |
| Analytic topology projection is necessary | PASS | Z82 removal causes violation RMSE up to 0.14454 |
| Object bridge dominates performance | NOT SUPPORTED | Z84 maximum object-RMSE effect 1.97e-5 |
| Compute matching | PASS | Z78 deployed BT 398,798 vs shared 399,990 parameters |

## What the model now is

The frozen method is **Stable Uncertainty-Calibrated BT-DPWM**: a damage-agnostic
robot expert; analytic topology projection; a block-triangular object expert; a
low-rank physical-context residual adapter with exact zero-context recovery;
nested support validation, hysteresis and permanent z=0 rollback; and conformal
intervals for physical-context estimation uncertainty.

The uncertainty roles are deliberately separated. Conformal intervals describe
physical-context estimation coverage. They do not decide rollout safety. The
empirically supported deployment safety chain is support validation + hysteresis
+ z=0 rollback + analytic projection.

## Reproducible artifacts

- `runs/g2_bt_dpwm_final_synthesis_v1/summary.json`: verdict and provenance.
- `runs/g2_bt_dpwm_final_synthesis_v1/claim_table.csv`: paper/table source.
- `runs/g2_bt_dpwm_final_synthesis_v1/budget_curves.csv`: numeric curve source.
- `runs/g2_bt_dpwm_final_synthesis_v1/completion_audit.json`: 40/40
  requirement-and-version-control checks passed.
- Z75--Z85 source summaries remain immutable inputs; Z78 retains failed runs.

## Next gate

Do not invent another architecture inside G2. If the narrow safe-adaptation story
is accepted, G3 starts only after the hardware readiness gate is green: serial,
calibrated eye-in-hand camera, measured markers and live pose, then intact/D3
low-amplitude smoke before the preregistered D2/D3 matrix. A superiority claim
requires new development and genuinely new confirmation seeds; 57/67 cannot be
reused for that purpose.
