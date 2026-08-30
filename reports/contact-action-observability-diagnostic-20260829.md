# Contact-Action Observability Diagnostic (2026-08-29)

## Purpose

After the H1 contact action-effect Gate failed, this diagnostic asks whether
candidate actions produce a learnable task signal at the planning timescale.
It trains no model and does not reopen or tune the failed low-rank operator.
The same 80 prefixes per robot, six actions per prefix, and three locks are
branched for H1, H5, and H10 simulator steps.

## Results

| Horizon | Robot | Within-prefix action fraction (position) | EE/object pairwise Spearman | Median XY candidate range | Median best-action margin | Final contact |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GenkiArm | 0.042% | 0.588 | 0.065 mm | 0.010 mm | 98.3% |
| 1 | Panda | 51.63% | 0.168 | 0.107 mm | 0.021 mm | 100% |
| 5 | GenkiArm | 0.520% | 0.604 | 0.726 mm | 0.123 mm | 75.8% |
| 5 | Panda | 51.39% | 0.374 | 1.013 mm | 0.198 mm | 100% |
| 10 | GenkiArm | 1.678% | 0.630 | 2.446 mm | 0.336 mm | 60.7% |
| 10 | Panda | 51.29% | 0.404 | 2.355 mm | 0.455 mm | 45.6% |

The H1 ranking target used by the failed Gate is poorly scaled for control,
especially on GenkiArm: candidate actions change the median best score by only
about 9.8 micrometres. At H10, both robots show millimetre-scale candidate
separation and positive analytic-EE/object distance association.

However, the longer horizon crosses a discrete contact boundary. By H10 only
60.7% of GenkiArm and 45.6% of Panda branches retain contact. A single smooth
response operator therefore averages incompatible contact-retained and
contact-lost dynamics. This explains why merely changing the regression head
is not a justified next step.

## Decision boundary

This diagnostic does **not** rescue the H1 mechanism and is not evidence of
closed-loop benefit. It permits exactly one new falsifiable hypothesis:

> Under a diagnosed joint lock, multi-step counterfactual action prediction is
> factorized into an explicit contact-mode survival process and a
> mode-conditioned cumulative response, while analytic projection continues to
> enforce the lock and state isolation protects published free-joint state.

Before implementation, this hybrid-mode hypothesis needs a novelty check
against switching/hybrid world models and contact-mode dynamics. Its first
Gate must use identical H10 candidate branches, a parameter-matched
non-factorized recurrent baseline, held-out middle locks on both arms, and
jointly require calibrated contact-mode prediction, improved object response,
improved action ranking, and lower regret. Failure stops the route; it cannot
be repaired by adding another head or changing horizon.

## Scope limitation

These are Push/contact branches only. They do not validate Grasp, visual
robustness, five untouched training seeds, cross-physics generalization, or
closed-loop MPC. They establish only that H10 has a nontrivial action signal and
that contact-mode switching is a necessary modeling target.
