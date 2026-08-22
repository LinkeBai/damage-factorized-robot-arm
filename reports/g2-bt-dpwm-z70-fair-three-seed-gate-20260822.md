# Z70 Stable BT-DPWM — Fair Three-Seed Gate

Z69 removes a hidden source of seed variance: the shared backbone was trained
with zero topology inputs, but earlier BT checkpoints fed real mask/angle values
through those untrained random columns. Z69 copies the same-seed shared robot
block, zeros only those columns, and retains analytic state/action projection.
Free-arm regression against shared falls from tens of percent to at most 1.23%
across seeds 7/17/27.

With the same adapter architecture, training domains, context encoder,
calibration transitions, and acceptance budget, the aggregate gains are:

| transitions | 0 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| BT own gain (%) | 0.00 | 0.21 | 1.38 | 3.53 | 7.24 |
| shared own gain (%) | 0.00 | 0.20 | 0.63 | 2.78 | 6.45 |
| BT relative to shared (%) | -0.73 | -0.72 | +0.04 | +0.03 | -0.19 |

BT has higher aggregate sample efficiency at K=10 and K=25 and is effectively
tied at K=50, while maintaining exactly zero locked-coordinate violation. Every
BT seed/domain/budget has non-negative gain relative to its own K=0 model.

Authoritative artifact:
`runs/g2_bt_dpwm_z69_adapter_z70/three_seed_fair_gate_v1/summary.json`.
