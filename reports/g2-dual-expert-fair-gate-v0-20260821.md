# Dual-Expert Fair Attribution — Gate V0

**Date:** 2026-08-21

**Decision:** NO-GO for factorization as an independently supported prediction
advantage; do not proceed to BT-DPWM

## Motivation

Earlier product-expert comparisons showed approximately 53% depth-10 free-arm
improvement over an ordinary projected ensemble. That comparison mixed a
20-epoch generic recurrent ensemble with a 60-epoch graph joint specialist, so
architecture, objective, and optimization budget were confounded with
factorization.

V0 isolates shared versus independent recurrence using the same graph model,
same data, same optimizer, same rollout loss, and the same deployment projection.

## Matched variants

| Variant | Hidden | Epochs | Parameters | Training objective |
|---|---:|---:|---:|---|
| Shared parameter-matched | 136 | 60 | 338,102 | normalized joint + object |
| Shared compute-matched | 96 | 120 | 169,542 | normalized joint + object |
| Independent joint | 96 | 60 | 169,542 | joint only |
| Independent object | 96 | 60 | 169,542 | object only |

The independent pair has 339,084 parameters, within 0.3% of the
parameter-matched shared model. Its two 60-epoch trainings match the 120 model-
epochs of the compute-matched shared control.

## Frozen gate

At depth 10, independent experts must improve free-arm RMSE by at least 10%
against both shared controls, regress at most 2% on object RMSE relative to the
best shared control, and retain zero constraint violation.

## Results

| Method | Depth-1 free | Depth-5 free | Depth-10 free | Depth-10 object | Violation |
|---|---:|---:|---:|---:|---:|
| Shared parameter-matched | 0.04 | 0.18 | 0.34 | 0.05 | 0 |
| Shared compute-matched | 0.05 | 0.18 | 0.32 | 0.05 | 0 |
| Independent experts | 0.05 | 0.19 | 0.36 | 0.04 | 0 |

Independent experts regress `3.26%` against the parameter-matched shared model
and `9.82%` against the compute-matched shared model on free-arm RMSE. They
improve object RMSE by `3.94%`, but the preregistered joint criterion fails.

## Interpretation

The earlier 53% result cannot be attributed to joint/object factorization. A
fairly trained shared graph model matches or exceeds independent recurrence on
free-arm prediction while retaining comparable object performance and exact
projection. The earlier gain was primarily a specialist-backbone/training-
protocol gain relative to the older generic ensemble baseline.

Independent training does produce a small object advantage, indicating some
task interference, but it is not a joint-and-object Pareto improvement and does
not support dual experts as the core method.

BT-DPWM is an upgrade of the dual-expert hypothesis. Under the frozen decision
sequence, its projected physical interface should not be trained after the base
factorization gate fails. The existing mixed-backbone dual-expert system remains
a useful engineering incumbent, but its performance must not be presented as
evidence that factorization is the causal innovation.
