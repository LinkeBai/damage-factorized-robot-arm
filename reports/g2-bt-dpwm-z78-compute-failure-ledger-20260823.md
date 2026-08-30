# BT-DPWM Z78 Compute and Failure Ledger

## Parameter fairness

The ledger instantiates the frozen classes and counts every module parameter.

| module | parameters |
|---|---:|
| shared h136 base | 338,102 |
| BT-DPWM base | 336,910 |
| projected residual adapter | 10,224 |
| uncertain context encoder | 51,664 |
| shared deployment total | 399,990 |
| BT deployment total | 398,798 |

BT is 0.298% smaller than the shared deployment stack. The comparison is thus
parameter matched without granting BT extra capacity. The frozen training
budgets are shared/scaffold 240 epochs, BT object block 120 epochs, shared and
BT adapters 200 epochs each, and uncertain encoder 500 epochs.

## Observed wall clock

The confirmation session ran on an RTX 4060 Laptop GPU (8 GiB), Warp 1.16.0,
CUDA Toolkit 12.9, and Intel Family 6 Model 183 CPU. Boundaries come from
captured process starts and adjacent artifact timestamps; they have about one
minute boundary uncertainty and are engineering measurements, not isolated
benchmark trials.

| stage | seed57 (min) | seed67 (min) |
|---|---:|---:|
| V0 fair baselines | 10.9 | 12.9 |
| Z32 base + scaffold | 27.1 | 27.5 |
| Z69 topology-column recovery | 7.1 | 7.1 |
| Z70 goal queries + both adapters | 24.4 | 25.2 |
| Z65 uncertain encoder | 0.8 | 0.9 |
| Z75 evaluation | 1.1 | 1.2 |
| total | 71.4 | 74.8 |

Warp/CUDA batch stepping and trajectory caches are active. The largest avoidable
serial cost is the Z70 collection of four CPU goal queries for each of 67
domains before GPU adapter epochs. A cloud GPU alone will not remove that CPU
optimizer bottleneck; parallel/batched goal-query generation is the relevant
engineering improvement.

## Retained failure chain

The machine ledger references ten non-passing run summaries rather than copying
only successful endpoints:

- Z71 five-seed safety failure;
- Z75 development strict shared-superiority failure;
- Z76 independent paired-equivalence failure;
- Z77 robustness strict-monotonicity failure;
- seed57/67 V0, Z32, and Z69 Y0 NO-GO summaries.

This does not mean every narrow claim failed. Z75 development safety, Z76
confirmation safety, and Z77 robustness safety each pass their stated sub-gates.
The top-level `passed: false` values preserve the stronger preregistered criteria.
In particular, Z69 is important recovery evidence but remains Y0 NO-GO because
object rollout is still worse than shared.

Authoritative artifact:

- `runs/g2_bt_dpwm_z78_compute_failure_ledger/summary.json`
