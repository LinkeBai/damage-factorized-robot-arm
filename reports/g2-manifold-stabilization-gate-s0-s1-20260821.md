# Constraint-Manifold Stabilization — Gates S0/S1

**Date:** 2026-08-21

**Decision:** NO-GO for FT-specific stabilization attribution

## Hypothesis

Product-space fusion may improve autonomous rollout by keeping recurrent states
on the damaged-system constraint manifold, rather than by making every local
joint prediction more accurate.

The attribution comparison includes ordinary autonomous ensemble, common-state
ensemble, ordinary plus direct analytic projection, a separately trained
matched graph joint expert, its projected version, and FT product fusion. The
primary endpoint is depth-10 free-arm RMSE on held-out D3.

## Frozen criteria

FT product fusion must:

- improve at least 5% over direct projection;
- regress at most 5% relative to the projected matched joint expert;
- regress at most 2% on object RMSE relative to ordinary autonomous rollout;
- gain at least 10 percentage points more at depth 10 than at depth 1.

S0 used seed 7. After its PASS, S1 froze seeds 7/17/27/37/47 and required 4/5
passing seeds. Evaluation stopped once two seeds failed, because 4/5 became
mathematically impossible.

## Results

| Seed | FT improvement vs direct projection | FT regression vs projected matched | Object regression | Depth-10 minus depth-1 gain | Decision |
|---:|---:|---:|---:|---:|---|
| 7 | +55.60% | -5.82% | -3.31% | +23.40 pp | PASS |
| 17 | +43.74% | +42.85% | -1.89% | +29.50 pp | NO-GO |
| 27 | +54.22% | +30.98% | -0.03% | +23.33 pp | NO-GO |

Negative regression means FT is better. Seeds 37/47 were not completed after
the aggregate gate became impossible.

## Interpretation

The broad stabilization mechanism is supported in all evaluated seeds:

- FT is 44%--56% better than ordinary direct projection at depth 10.
- The FT advantage over ordinary rollout grows by 23--30 percentage points from
  depth 1 to depth 10.
- Object prediction is preserved or improved.

However, the effect is not attributable to FT fixed-transform geometry. A
separately trained matched graph joint expert plus the same analytic projection
beats FT by 43% and 31% in seeds 17 and 27. Therefore the evidence supports
**modular joint/object rollout plus constraint projection**, not an FT-specific
method contribution.

Directly projecting the ordinary full-state ensemble is insufficient, showing
that merely zeroing locked coordinates does not explain the result. A separate
joint dynamics expert is the important factor; whether fixed-transform geometry
adds stable value remains unsupported.

## Decision

Stop S1 and do not claim FT-specific constraint-manifold stabilization as the
core innovation. The remaining positive result is a modular architecture:
separate joint and object experts, with exact projection on the joint expert.
Before treating that as a paper method, novelty must be strengthened beyond
generic task decomposition and compared with standard modular/factorized world
models.
