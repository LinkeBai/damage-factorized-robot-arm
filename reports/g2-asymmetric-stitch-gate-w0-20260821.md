# Asymmetric Subspace Stitching — Gate W0

**Date:** 2026-08-21

**Decision:** NO-GO; do not expand beyond seed 7

## Hypothesis

V0 found that the shared compute-matched graph had the best free-arm prediction,
while the independent object specialist had the best object prediction. W0
tests a zero-training asymmetric composition using the shared graph for joint
coordinates and the independent specialist for object coordinates, with
separate recurrent hidden states and exact projection.

## Frozen gate

Relative to the shared compute-matched model, asymmetric stitching must keep
free-arm regression within 2%, improve object RMSE by at least 10%, and improve
overall RMSE by at least 5%. It must also improve overall RMSE by at least 5%
relative to fully independent experts and maintain zero constraint violation.

## Results

| Method | Depth-10 free | Depth-10 object | Depth-10 overall | Violation |
|---|---:|---:|---:|---:|
| Shared compute-matched | 0.32 | 0.05 | 0.25 | 0 |
| Independent experts | 0.36 | 0.04 | 0.27 | 0 |
| Asymmetric stitch | 0.32 | 0.04 | 0.24 | 0 |

Asymmetric stitching improves free-arm RMSE by `1.32%`, object RMSE by `3.43%`,
and overall RMSE by `1.34%` relative to the strongest shared control. It improves
overall RMSE by `10.05%` relative to fully independent experts.

## Interpretation

The zero-training stitch recovers the strongest coordinate block from each
provider and is a Pareto improvement over fully independent experts. However,
the gain over the strongest fairly trained shared model is too small to support
a new method. The object specialist's apparent V0 advantage shrinks from the
aggregate comparison when evaluated inside the asymmetric recurrent feedback
loop.

Subspace stitching remains a useful engineering ensemble but not a justified
core innovation under the preregistered thresholds. Do not tune routing weights
or expand seeds after observing W0.
