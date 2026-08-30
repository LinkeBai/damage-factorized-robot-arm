# Gate N1 Result: Capsule–Box Multi-Contact Oracle

**Decision:** NO-GO before learned-model training

| Metric | Threshold | Result |
|---|---:|---:|
| Exact-record negative normal fraction | <= 1% | 0.40% |
| Multi-contact transition fraction | diagnostic | 13.68% |
| Candidate projection relative RMSE | <= 20% | 27.41% |

N1 replaced the N0 bounding-circle basis with analytic planar closest points
between the tool/pusher capsule segments and the axis-aligned block box. Data,
impulse constraints, optimizer, projection budget and thresholds were held
fixed. Projection error increased from `24.13%` to `27.41%`.

The mismatch therefore cannot be repaired by replacing a center normal with a
planar box-face normal. MuJoCo resolves three-dimensional capsule/box contacts,
penetration, corners and simultaneous table constraints; the current
two-dimensional candidate reconstruction is not an adequate deployment basis.

No learned set operator, contact detector, five-seed expansion or G3 experiment
is authorized. Further hand-written contact geometry would amount to simulator
replication rather than a validated damage-dynamics mechanism.
