# Tangent-Manifold Graph — Gate T0

**Date:** 2026-08-21

**Decision:** NO-GO; do not expand beyond seed 7

## Hypothesis

A full-chain joint graph may gain additional stability by retaining locked nodes
as spatial message relays while removing their temporal hidden state and learned
joint increments. This explicitly restricts recurrent dynamics to the tangent
space of the damage-conditioned constraint manifold.

## Controls

- `projected_matched`: independent graph joint expert trained without the damage
  mask, followed by exact projection;
- `topology_projected`: full-chain graph trained with true mask, projected input,
  projected action, and projected output, but unrestricted locked-node temporal
  hidden state;
- `tangent_manifold`: same initialization and capacity as `topology_projected`,
  additionally zeroing locked-node recurrent hidden state and joint delta.

All variants use the same frozen ordinary ensemble as object expert.

## Frozen seed-7 gate

At depth 10, tangent dynamics must improve free-arm RMSE by at least 5% over both
controls, preserve object RMSE within 2%, and maintain zero constraint violation.

## Results

| Method | Depth-1 free RMSE | Depth-5 free RMSE | Depth-10 free RMSE | Depth-10 object RMSE | Violation |
|---|---:|---:|---:|---:|---:|
| projected matched | 0.04 | 0.20 | 0.44 | 0.48 | 0 |
| topology projected | 0.05 | 0.22 | 0.49 | 0.48 | 0 |
| tangent manifold | 0.05 | 0.22 | 0.51 | 0.48 | 0 |

Tangent dynamics regresses `17.14%` relative to projected matched and `4.69%`
relative to topology projected. Object regression is only `0.46%`, and exact
constraints hold, so the failure is specifically free-joint prediction.

## Interpretation

Removing locked-node temporal memory is not a useful tangent/normal
decomposition in this architecture. Although locked coordinates cannot move,
their recurrent features can still encode damage-conditioned history useful to
neighboring free-joint prediction or local state estimation. Zeroing that memory
removes information without improving constraint satisfaction, because exact
output projection already eliminates normal-coordinate error.

The result also shows that feeding the topology mask during joint-expert
training does not beat the simpler damage-agnostic matched expert plus analytic
projection in seed 7. The proposed tangent-space mechanism therefore does not
provide the missing innovation and must not be expanded or renamed as a positive
result.
