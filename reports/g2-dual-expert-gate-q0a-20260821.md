# G2 Dual-Expert World Model — Gate Q0-A

**Date:** 2026-08-21

**Decision:** TWO-SEED PASS

**Config:** `config/experiment/g2_dual_expert_gate_q0a_v1.yaml`

## Question

Can a frozen FT-GWM K1 structural expert provide the joint state while a
three-member ordinary, constant-condition ensemble provides the object state,
without degrading either prediction subspace?

No fusion gate or joint fine-tuning is used. At each rollout step every
predictive member receives the same fused state returned at the preceding step.

## Frozen gate

- Primary domain: `D3__mixed_composition` (D3 absent from training).
- Object RMSE regression relative to ordinary ensemble: at most 2%.
- Free-arm RMSE regression relative to ordinary ensemble: at most 5%.
- Constraint violation RMS: at most `1e-7`.
- Initial confirmation seeds: 7 and 17.

## Results

| Seed | Ordinary object | Fusion object | Object change | Ordinary free arm | Fusion free arm | Free-arm change | Violation | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 0.3103 | 0.3036 | -2.15% | 0.5728 | 0.2643 | -53.86% | 0 | PASS |
| 17 | 0.1467 | 0.1438 | -1.98% | 0.4977 | 0.2932 | -41.08% | 0 | PASS |

Artifacts:

- `runs/g2_dual_expert_gate_q0a/seed7_v1/summary.json`
- `runs/g2_dual_expert_gate_q0a/seed17_v1/summary.json`
- Each run also contains `results.csv`, `ordinary_ensemble.pt`, and `ft_gwm.pt`.

## Interpretation boundary

Q0-A establishes engineering and prediction-fidelity feasibility only. The
small object improvement is consistent with conditioning the object expert on a
better joint trajectory, but is not yet a separately supported mechanism claim.
The result does not establish that cross-expert discrepancy detects consensus
failures, improves fixed-depth risk ranking, or improves MPC control.

The next admissible experiment is Q0-B: evaluate whether structural discrepancy
adds out-of-sample error information conditional on ordinary ensemble
uncertainty and whether it improves fixed-depth AURC by the frozen threshold.
