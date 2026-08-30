# G2-R0 free-joint gate miss diagnosis

## Scope

This is a read-only analysis of the frozen seed-7 ICRA audit rows. It does not
change the model, threshold, context rule, adapter, trajectories, or any frozen
artifact. The audited cell is `D3__mixed_unseen`, H50, with 30 trajectories and
three non-overlapping terminal windows per trajectory.

## Result

Across the 90 paired terminal windows, the matched shared baseline has
free-joint RMSE 0.7622 and IPWM has 0.8080, a 6.01% regression under direct
recomputation from raw squared errors. The previously reported aggregate is
5.97%; the small difference is attributable to the aggregation path and does
not change the gate verdict. The trajectory-cluster 95% interval for the paired
MSE increase is [0.06885, 0.07538], and 93.3% of terminal windows have a
positive MSE delta. This is a systematic miss, not one outlier trajectory.

| Window start | Shared free RMSE | IPWM free RMSE | Free change | Object improvement | Pusher change |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.2677 | 1.3482 | +6.35% | +33.93% | +0.64% |
| 50 | 0.3057 | 0.3116 | +1.94% | +36.43% | +6.75% |
| 100 | 0.2057 | 0.2096 | +1.90% | +46.31% | +0.47% |
| all | 0.7622 | 0.8080 | +6.01% | +36.49% | +3.19% |

The excess is concentrated in the first H50 window: its paired MSE increase is
0.21063, compared with 0.00366 and 0.00162 in the later windows. The same cell
still improves object RMSE by 36.49%, while pusher RMSE changes by 3.19%.

## Interpretation boundary

The failure cannot be described as random noise or as a single adverse trial.
It is a long-horizon robot-state side effect concentrated in the initial
contact/approach segment. At the same time, it does not invalidate the primary
object result and is not equivalent to a locked-coordinate safety violation:
analytic lock violation remains exactly zero.

The paper may claim selective object-dynamics improvement with a disclosed
free-state trade-off. It may not claim full-state non-regression, complete
dynamic recovery, or a universally safe learned rollout. Task-level impact
must be evaluated separately in a frozen closed-loop protocol.

## Required next experiment

1. Export per-joint, per-depth squared errors for this cell without modifying
   the frozen ICRA audit artifacts.
2. Identify whether the excess lies in position, velocity, or both, and whether
   it projects to material pusher error.
3. Run the frozen closed-loop simulation protocol with shared and IPWM models.
4. If task outcomes are unaffected, report this as a state-prediction boundary.
   If task outcomes degrade, design a free-state safeguard on development seeds
   and evaluate it only on new confirmation seeds.

## Reproduction

```powershell
python scripts/analyze_g2_r0_free_joint_failure.py `
  --input runs/g2_r0_icra_audit_20260824/seed7/raw_window_metrics_30traj.json `
  --output runs/g2_r0_icra_audit_20260824/seed7/free_joint_failure_diagnosis.json
```

Machine-readable output:
`runs/g2_r0_icra_audit_20260824/seed7/free_joint_failure_diagnosis.json`.
