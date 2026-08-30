# Gate N0 Result: Set-Valued Multi-Contact Oracle

**Decision:** NO-GO before learned-model training

| Metric | Threshold | Result |
|---|---:|---:|
| Exact-record negative normal fraction | <= 1% | 0.40% |
| Multi-contact transition fraction | diagnostic | 13.68% |
| Candidate projection relative RMSE | <= 20% | 24.13% |

The immediate contact snapshot and orientation convention are valid, and the
data contain a material multi-contact subset. However, the tool/pusher
bounding-circle candidate frames cannot represent the measured aggregate
impulse within the preregistered tolerance. The learned set operator was not
authorized.

An earlier numerical run initialized `softplus(0)=0.693` against impulses of
approximately `0.0025 N*s` and was discarded as a solver-scale error. The
reported result uses a scale-correct `-8` raw-normal initialization under the
same frozen 400-step projection budget.
