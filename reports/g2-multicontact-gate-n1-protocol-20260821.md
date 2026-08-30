# Gate N1 Protocol: Capsule–Box Multi-Contact Oracle

**Status:** frozen before execution

Gate N0 used block-center/bounding-circle normals and failed with `24.13%`
relative projection RMSE against a `20%` threshold. N1 changes only candidate
geometry: it computes planar closest points between each tool/pusher capsule
segment and the actual axis-aligned block box. Impulse constraints, data,
optimizer, budget and thresholds remain unchanged.

Passing N1 authorizes a seed-7 learned set-valued impulse operator. Failure
stops the multi-contact branch before model training.
