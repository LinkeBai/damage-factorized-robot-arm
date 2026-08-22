# BT-DPWM Z79 Counterfactual Uncertainty Calibration

## Protocol

The Z75 decision rule is unchanged. For every observable context proposal, Z79
additionally evaluates the best proposed context on the same independent goal
rollouts even when the gate rejects it. This provides a counterfactual harm
target without feeding evaluation results back into acceptance. Seven seeds
(7/17/27/37/47/57/67) yield 24 non-degenerate proposals.

Candidate harm is the percentage increase in rollout RMSE relative to the
previous-budget incumbent. Positive values are harmful. Posterior uncertainty is
the mean standard deviation emitted by the Z65 physical-context encoder.

## Result

Three of 24 proposals are harmful. The nested support/hysteresis gate rejects
all three; none of the 11 accepted proposals is harmful. Thus the empirical
safety evidence belongs to support validation and hysteresis.

Posterior spread does not rank rollout risk. Overall Spearman correlation between
mean posterior standard deviation and candidate harm is -0.734 (p=4.47e-5), the
opposite of the required direction. At K50, where all three harmful candidates
occur, correlation is +0.119 (p=0.779). Topology-stratified results are D2 -0.429,
D3 -0.750, and D4 -0.550; only D3 is significant, again in the wrong direction.

| lowest-uncertainty coverage | retained | harmful fraction | mean harm (%) | worst harm (%) |
|---:|---:|---:|---:|---:|
| 25% | 6 | 16.7% | -1.154 | +0.024 |
| 50% | 12 | 25.0% | -2.265 | +2.235 |
| 75% | 18 | 16.7% | -4.971 | +2.235 |
| 100% | 24 | 12.5% | -6.892 | +2.235 |

Lower-uncertainty retention does not reduce harmful frequency monotonically and
actually removes many of the most beneficial proposals. This agrees with the
Z76 threshold ablation: the only proposal rejected solely for std>0.30 was a
beneficial seed67 D3-unseen update.

## Decision

The uncertainty calibration gate fails. Z65 posterior spread may describe
physical-context ambiguity or shift magnitude, but it is not a calibrated
probability or ranking of task-rollout harm. The method cannot currently be
marketed as uncertainty-calibrated safe adaptation.

The BT-DPWM mechanism is not abandoned: analytic projection, block-triangular
factorization, support validation, hysteresis, and z=0 fallback retain their
evidence. The next allowed improvement is development-only recalibration of the
same posterior against held-out context error/risk, followed by new untouched
confirmation seeds. Seeds 57/67 cannot validate a rule designed from this audit.

Authoritative artifact:

- `runs/g2_bt_dpwm_z79_uncertainty_counterfactual/calibration_v1/summary.json`
