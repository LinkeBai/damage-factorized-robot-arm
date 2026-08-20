# Gate I: Reduced-Coordinate Graph World Model

**Date:** 2026-08-20
**Seed-7 decision:** **PASS; expand to five seeds**

## Method

RC-GWM removes diagnosed locked joints from the dynamic coordinate graph,
connects their nearest free neighbors, zeros recurrent state at removed nodes,
and pools only active nodes for object prediction. It predicts free joint
coordinates and reconstructs the full state analytically, giving exact lock
satisfaction during both training and rollout.

This differs from post-hoc projection: the learned transition is optimized in
the same reduced coordinate system used at inference.

## Frozen seed-7 result: `D3__mixed_composition`

| method | parameters | overall RMSE | free-arm RMSE | object RMSE | violation RMS |
|---|---:|---:|---:|---:|---:|
| matched graph | 299,782 | 0.1712 | 0.2121 | 0.0306 | 0.152518 |
| matched direct projection | 299,782 | 0.1614 | 0.2123 | 0.0321 | 0 |
| RC-GWM | 299,526 | 0.1586 | 0.2095 | 0.0153 | 0 |

Relative to matched graph, RC-GWM improves object RMSE by **50.00%** and
free-arm RMSE by **1.22%**, with exact zero violation. It passes the frozen
Gate I requirement that neither object nor free-arm regress by more than 5%.

D2 and D4 mixed-composition evaluations also improve both metrics. On the
harder `D3__mixed_unseen` domain, object RMSE improves while free-arm RMSE
regresses; the five-seed expansion must report this boundary separately.

## Next gate

Run seeds 17, 27, 37, and 47 without architecture or threshold changes.
Report per-domain seed bootstrap intervals. A main-method claim requires the
primary D3 composition to retain zero violation and non-regression in both
object and free-arm aggregate metrics; the unseen-physics domain is a required
failure-boundary analysis.

## Artifacts

- `src/robotarm/models/reduced_coordinate_graph.py`
- `config/experiment/g2_reduced_coordinate_gate_i_v1.yaml`
- `runs/g2_reduced_coordinate_gate_i/seed7_v1/summary.json`
- `runs/g2_reduced_coordinate_gate_i/seed7_v1/results.csv`
