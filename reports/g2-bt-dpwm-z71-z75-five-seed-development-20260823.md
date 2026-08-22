# BT-DPWM Z71--Z75 Five-Seed G2 Development Audit

## Frozen expansion

Seeds 37 and 47 were committed before their checkpoints existed, extending the
existing 7/17/27 set without result-based replacement. Shared and BT used the
same calibration trajectories, adapter architecture, optimization budget, and
evaluation targets.

## What failed

The first five-seed Z71 evaluation retained a positive K50 mean BT-own gain
(5.72%, seed-bootstrap 95% CI 3.55--8.06%) and zero constraint violation, but
failed safety: seed47 lost 9.96% on D3 composition and 5.46% on D3 unseen after
an accepted context. BT versus shared at K50 was -0.43% (95% CI -1.14--0.34%).

Code audit found that the first uncertain proposal bypassed replacement
hysteresis, z=0 was not a candidate after the first acceptance, and only the
latest validation window was remembered. Thus the implementation did not fully
realize the claimed reversible safe fallback.

## Same-mechanism correction

Z75 requires 15 fit transitions for D3/D4, applies the same hysteresis to first
and replacement proposals, rejects mean context standard deviation above 0.30,
keeps z=0 permanently available, and requires every candidate to avoid more
than 1% regression on all accumulated nested support-validation windows.

| transitions | 0 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| BT own gain (%) | 0.000 | 0.504 | 0.504 | 3.357 | 3.357 |
| shared own gain (%) | 0.000 | 0.548 | 0.548 | 3.377 | 3.377 |

All five seeds, four domains, and five budgets are non-negative relative to the
same BT K0 checkpoint. The aggregate curve is monotonic and locked-coordinate
violation is exactly zero. K25/K50 BT-own gain has a seed-bootstrap 95% CI of
approximately 1.36--5.61%, excluding zero.

The strict preregistered sample-efficiency sign gate does not pass: BT trails
shared by 0.020 percentage points at K25. K50 BT relative to shared is -0.778%,
inside the preregistered -1% engineering tolerance but not a superiority claim.

## Status

Z75 is a development safety/monotonicity pass, not a final G2 Go, because its
rules were produced from the five-seed safety audit. Independent confirmation
seeds are required. Z71 and intermediate Z72--Z74 results remain preserved as
failure/development evidence.

Authoritative artifacts:

- `runs/g2_bt_dpwm_z71_five_seed/five_seed_gate_v1/summary.json`
- `runs/g2_bt_dpwm_z75_nested_support/five_seed_development_v1/summary.json`
