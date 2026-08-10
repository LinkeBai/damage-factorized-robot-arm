# G1 Six-DoF Provisional Gate

Date: 2026-08-07

Plan: `PROJECT-PLAN-V4.md`

Decision: **NOT READY / NO GO YET**

This is an offline engineering checkpoint. G1 formally depends on G0 measured
geometry and residual ranges, which are still unavailable.

## Completed infrastructure

- Six-joint simple and seven-mesh MuJoCo models.
- Six-dimensional mask/action and 12-dimensional proprioceptive state.
- Conditional recurrent world model with stochastic prior/posterior latent.
- Latent optimization with frozen world model.
- Four prediction methods:
  - topology-only;
  - residual-only;
  - monolithic parameter-matched descriptor;
  - factorized topology + residual (DFWM).
- Immutable D2/D3 composition split and provisional Reach target split.
- Frozen CEM-MPC deployment path.
- Runtime, memory, parameter-count and adaptation-time logging.

## Current smoke observations

The 10-epoch four-method RSSM smoke is pipeline validation only. Residual-only
currently has lower held-out prediction error than factorized DFWM, so there
is no factorization claim.

Frozen-MPC D3 smoke over two evaluation targets:

| Method | Calibration | Success | Final distance |
|---|---:|---:|---:|
| Topology-only MPC | 0 | 0/2 | 0.374 m |
| DFWM MPC | 1 trajectory | 0/2 | 0.349 m |

DFWM improves final distance slightly, but neither method succeeds. This fails
the control-recovery gate and must not be reported as G1 success.

## Next simulation actions

1. Stabilize intact Reach with frozen MPC before adaptation comparisons.
2. Add multi-step rollout loss and reward/continue supervision.
3. Tune planner horizon/candidates only on the validation target split.
4. Run D2/D3 K=0/1/2/5 only after the intact baseline reaches repeatably.
5. Use G0 measurements to replace provisional residual ranges and target split.
