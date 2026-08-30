# Gate H: Low-Capacity Gated Reaction Audit

**Date:** 2026-08-20
**Corrected decision:** **PROVISIONAL PASS**, superseded by Gate I

## Question

Can an exact lock projection plus a low-capacity gated reaction head preserve
the prediction fidelity of a `hidden=128` matched graph model? Gate H passes
only if constraint violation is approximately zero and both object and
free-arm RMSE regress by no more than 5% relative to the matched graph.

## Frozen setup

- seed: 7
- train topologies: intact, D2, D4; primary held-out topology: D3
- epochs: 20 base + 20 adapter; rollout horizon: 10
- matched graph hidden size: 128
- gated reaction bottleneck: 16; gate logits initialized to -4
- gated reaction trainable parameters: 2,744
- comparison methods: matched graph, matched direct projection,
  unconstrained residual adapter, gated reaction plus exact projection

The free-arm metric uses the true damage mask for every method. Locked joint
position and velocity are excluded even when topology is hidden from the
predictor.

## Primary result: `D3__mixed_composition`

| method | parameters | overall RMSE | free-arm RMSE | object RMSE | violation RMS |
|---|---:|---:|---:|---:|---:|
| matched graph | 299,782 | 0.1712 | 0.2121 | 0.0306 | 0.152518 |
| matched direct projection | 299,782 | 0.1614 | 0.2123 | 0.0321 | 0 |
| unconstrained residual | 515,469 | 0.1795 | 0.2120 | 0.0192 | 0.176171 |
| gated reaction + projection | 302,526 | 0.1612 | 0.2127 | 0.0220 | 0 |

Relative to matched graph, gated reaction has:

- constraint violation: **0**
- object RMSE regression: **-27.93%** (an improvement)
- free-arm RMSE regression: **+0.29%**

The original report compared the matched graph's all-joint arm RMSE against
the gated model's free-joint RMSE. After applying the true damage mask to both
methods, Gate H meets its frozen primary-domain criteria. This correction was
discovered while implementing Gate I. Gate H is not expanded because the
reduced-coordinate model provides a simpler, end-to-end alternative with
cleaner attribution and fewer parameters.

## Audit correction to Gate G

The earlier Gate G direct-projection run was not capacity matched: the runner
assigned `hidden=96` (169,542 parameters) to `graph_matched_projected` while
the reference graph used `hidden=128` (299,782 parameters). Its reported large
degradation is invalid as evidence against direct projection.

After fixing the runner, matched direct projection is close to gated reaction
on overall and free-arm RMSE and satisfies the exact constraint. The reaction
head's clearest incremental signal is lower object RMSE, but Gate H does not
establish that signal across seeds and the pre-registered stop rule forbids a
five-seed expansion.

## Decision and next direction

Do not treat the earlier NO-GO as valid. Nevertheless, prefer Gate I's
reduced-coordinate transition over CR-GWM: it enforces feasibility by
construction rather than through an additional reaction adapter and can be
trained end to end. Gate H remains an informative ablation.

## Artifacts

- `config/experiment/g2_gated_reaction_gate_h_v1.yaml`
- `src/robotarm/models/gated_reaction_graph.py`
- `runs/g2_gated_reaction_gate_h/seed7_v2/summary.json`
- `runs/g2_gated_reaction_gate_h/seed7_v2/results.csv`
