# Gate M1b Result: Exact Contact Impulse Decomposition

**Decision:** NO-GO before learned-model training

## Oracle Results

| Metric | Threshold | Result |
|---|---:|---:|
| Contact transitions | >= 20 | 914 / 3600 |
| Implicit momentum reconstruction RMSE | <= 0.001 | 0.002949 |
| Negative normal pusher impulse fraction | <= 5% | 29.32% |
| Nominal friction-cone violation fraction | diagnostic | 61.05% |

Immediate post-`mj_step` force snapshots, separate tool/pusher and table
impulses, and analytic slide-joint damping reduce the force-balance error
substantially, but the preregistered oracle gate still fails.

## Interpretation

The remaining failure is representational. The deployable approximation uses
one closest-point line-segment normal, while MuJoCo can simultaneously contact
the block through the tool capsule, pusher capsule, edges, and table. Aggregated
world-frame impulse cannot in general be expressed by one unilateral normal
and its tangent. Training a larger network would not repair this mismatch.

## Decision Boundary

M1b learned pusher-impulse training is not authorized. Geometric contact-mode
detection and five-seed expansion remain blocked. A future attempt would need
per-contact candidate geometry and set-valued impulse aggregation, which is a
new method and protocol rather than a continuation of Gate M1b.
