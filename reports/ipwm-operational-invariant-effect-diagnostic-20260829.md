# IPWM Operational-Invariant Effect Diagnostic (2026-08-29)

## Hypothesis

The earlier shared raw-history and kinematic `J a` representations are embodiment-dependent. A new analytic observable maps intact and lock-reduced action responses into the current contact frame, whitens them by operational-space mobility, and normalizes them by joint-space control energy. This produces a 16-dimensional quantity invariant to padded DoF, joint coordinate scale and raw actuator magnitude. It uses only current model state, diagnosed lock and candidate action; robot identity, future state, solver force and contact outcome are forbidden.

For free DoFs `f`, the core terms are

\[
a_f=J_f M_{ff}^{-1}\tau_f,\quad
W_f=J_fM_{ff}^{-1}J_f^\top,\quad
e_f=\tau_f^\top M_{ff}^{-1}\tau_f,
\]

followed by contact-frame projection and whitening with the directional diagonal of `W_f`. The feature includes intact response, lock-reduced response, their normalized difference and energy ratio.

## Frozen diagnostic

- Dataset: the existing 2,880-row H10 exact-state GenkiArm/Panda counterfactual set.
- Training: 70% grouped prefixes and non-middle locks.
- Test: disjoint prefixes and the held-out middle lock on each arm.
- Estimator: identical closed-form Ridge for original observables and original plus invariant features.
- Go: at least 2/3 split seeds must improve both robots, pooled RMSE, Spearman by at least +0.10 and top-1 regret.

## Result

| seed | Genki RMSE baseline→invariant | Panda RMSE baseline→invariant | pooled gain | both arms | ΔSpearman | regret baseline→invariant |
|---:|---:|---:|---:|:---:|---:|---:|
| 7 | 0.11138→0.10149 | 0.59889→0.45390 | +23.65% | Yes | -0.0289 | 0.24308→0.22511 |
| 17 | 0.14038→0.14560 | 0.61796→0.51166 | +16.05% | No | +0.0518 | 0.21756→0.17907 |
| 27 | 0.12383→0.11925 | 0.57288→0.45781 | +19.28% | Yes | +0.0122 | 0.17084→0.17297 |

The observable has a strong pooled prediction signal and improves both arms in 2/3 splits, unlike raw shared history. It nevertheless passes **0/3** joint control-relevance gates: no split reaches the frozen +0.10 Spearman requirement, seed 7 worsens correlation and seed 27 worsens regret.

## Decision

**NO-GO as the new paper mechanism.** It may be retained as a mechanistic diagnostic showing that effective-mass/energy normalization reduces embodiment-dependent prediction error. It cannot support action-ranking, closed-loop, cross-task or 4+/5 claims. No feature deletion, alternative whitening, Ridge alpha, split or threshold will be tuned after inspection.

Artifacts: `scripts/diagnose_operational_invariant_effect.py` and `runs/operational_invariant_effect_v1/seed{7,17,27}.json`.
