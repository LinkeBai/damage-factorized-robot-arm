# Gate M v1 Result: Hybrid Contact Impulse

**Decision:** NO-GO for the total-velocity-change impulse proxy

## Frozen Result

| Metric | Result |
|---|---:|
| Training contact transitions | 916 / 3600 |
| Oracle velocity reconstruction RMSE | 0.000000 |
| Continuous operator rollout RMSE | 0.133269 |
| Hybrid proxy-impulse rollout RMSE | 0.337359 |
| Hybrid relative change | -153.14% |

The data contain ample contact, but the constrained hybrid model is worse than
the matched continuous operator under the same 60-epoch budget. Gate M2 v1
fails its preregistered 30% improvement threshold.

## Root Cause

The v1 target used total object velocity change as if it were pusher contact
impulse. On contact transitions, `47.49%` of these vectors project negatively
onto the approximate contact normal and `71.18%` violate the nominal friction
cone. The target combines pusher impulse, table friction, other simultaneous
contacts, and integration effects, so the unilateral impulse model cannot
represent it.

Direct MuJoCo pusher/tool-to-block impulse extraction reduces the ambiguity but
does not explain all state change: the remaining velocity-change RMSE is
`0.033050`. The next scientifically valid experiment would need exact
per-contact frames and a separate table/free-motion operator.

Post-hoc instrumentation corrected the sampling point to immediately after
`mj_step`, reducing the tool-only unexplained RMSE to `0.010691`. This does not
change the frozen M2 v1 decision; it motivated the separately preregistered
M1b oracle gate.

## Decision Boundary

This result does not validate a learned contact detector and does not authorize
five seeds or real-robot G3. It also does not refute event-driven contact in
general; it refutes the specific v1 proxy-label formulation. Any M1b attempt
must be preregistered as a new mechanism, not reported as tuning M2 v1.
