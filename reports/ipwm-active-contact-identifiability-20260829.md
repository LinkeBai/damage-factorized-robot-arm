# IPWM Active-Contact Identifiability Diagnostic (2026-08-29)

## Question

Does the original IPWM premise still have a viable cross-structure path: can a
short, deployable active-contact history reveal the residual physics needed to
predict H10 candidate responses under a held-out lock and held-out physical
composition?

## Data integrity

The valid dataset is
`runs/ipwm_active_contact_identifiability_v1/dataset_v2_seed20260829.npz`.
It contains 14,400 rows, exactly balanced between calibrated GenkiArm and the
official Panda model. Each of 80 prefixes per arm is evaluated under five
physics profiles, three locks, six candidate actions, an eight-step fixed probe,
and an H10 candidate branch. Lock violation is zero.

The earlier file `dataset_seed20260829.npz` is invalid and superseded because
unprefixed history names overwrote current candidate fields. It is retained for
audit only and must never be used for training or tables.

## Diagnostic comparison

A fixed closed-form Ridge probe compares current state, ordered K8 history,
equal-dimensional permuted history, and oracle physics values. Training excludes
the middle locks and held-out mixed profile; testing jointly holds out prefix,
mixed physics, GenkiArm j3, and Panda joint4.

| Split seed | Pooled RMSE gain from ordered history | Spearman gain | Lower regret | Both arms improve | Ordered beats permuted |
|---:|---:|---:|:---:|:---:|:---:|
| 7 | 35.02% | +0.384 | Yes | No | Yes |
| 17 | 38.43% | +0.556 | Yes | No | Yes |
| 27 | 41.28% | +0.454 | Yes | No | Yes |

The ordered sequence carries real task-response information: its improvements
are large and are absent after permutation. However, every shared split improves
Panda while regressing GenkiArm RMSE. It therefore passes **0/3** under the
frozen requirement that both structures benefit.

## Upper-bound interpretation

Separate robot-specific Ridge decoders were run only as an information upper
bound and are not deployable candidates. They show the opposite asymmetry:
history improves GenkiArm substantially, while current state alone is better on
Panda. This means neither “history contains no signal” nor “one universal raw
history decoder works” is supported. The response information is embodiment-
dependent under the present observables and probe.

Direct decoding of the three simulator physics scalars from history is worse
than the training-mean constant on the held-out composition in all splits. The
history therefore should not be described as recovering damping, friction, and
actuator scale; at most it encodes local response signatures.

## Decision

**No-Go for the current simulation mechanism route.** The preregistered rule
explicitly says that failure to improve held-out response on both robots must
stop the route rather than authorize another encoder or head. A graph encoder,
GRU, contrastive objective, per-robot normalization, or changed probe introduced
after these results would be post-hoc component search.

This result does not invalidate SI-IPWM's narrow single-arm state-isolation
claim. It does invalidate the planned escalation from that claim to a unified
cross-structure, contact-aware, control-relevant IPWM using the evidence and
hypotheses currently available.

## Publication consequence

The simulation-only paper remains approximately 3.2--3.4/5. The requested ICRA
4+/5 package cannot professionally proceed to the dual-arm Push/Grasp five-seed
confirmation matrix because no core mechanism has passed its development Gate.
Continuing requires genuinely new scientific input--for example a formal
embodiment-invariant observable derived independently of these failed Gates--or
a user-approved change of scope/claim. Repackaging, tuning, or adding Grasp data
cannot repair the missing causal/control mechanism.
