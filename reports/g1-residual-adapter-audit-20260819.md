# G1 Residual Adapter Audit (2026-08-19)

## Decision

Stop the current residual-correction and residual-FiLM branches before multi-seed
training. Neither branch produced fidelity-stable, direction-consistent gains on
D2 and D3. G1 remains No-Go for the claim that few-shot residual identification
improves Push prediction.

## Experiments

| Method | Fidelity | D2 multi-step | D3 multi-step | Mean | Decision |
|---|---:|---:|---:|---:|---|
| Grouped correction, bounded, one-step | full | -1.34% | +1.55% | +0.09% | stop |
| Static FiLM, rank 4, scale 0.10 | medium | -0.89% | -1.21% | -1.05% | stop |
| Dynamic FiLM, rank 4, scale 0.05 | medium | +1.50% | -0.51% | +0.48% | stop |
| Dynamic FiLM, rank 4, scale 0.10 | medium | +0.72% | -0.73% | -0.02% | stop |

Positive percentages mean lower multi-step RMSE than topology-only. Short
screening runs were used only for direction selection; medium/full runs made the
gate decision.

## Interpretation

True simulator residual descriptors do not provide a strong, stable advantage
under the current Push protocol and frozen topology dynamics. Output correction
over-corrects autonomous rollouts. Static hidden modulation is too coarse.
State/action-conditioned modulation improves D2 in short runs but does not
generalize consistently to D3 as fidelity increases.

This agrees with the earlier oracle upper-bound result: residual inference is not
the only bottleneck. More K, a larger history encoder, or more seeds cannot fix
an absent mechanism-level advantage.

## Next Gate

Follow Plan V5's pivot instead of tuning this branch further: evaluate robust
zero-shot structured dynamics and uncertainty-aware control. Any later residual
method must first pass a cheap oracle test on both D2 and D3 before latent
inference or multi-seed experiments are allowed.
