# G1 overnight method audit (2026-08-19)

## Scope

This audit ran the remaining method branches continuously under the corrected,
goal-directed Push protocol. It used D2/D3, exact contact coverage, active
calibration probes, disjoint evaluation trajectories, and staged gates to stop
failed branches before expensive 5-seed expansion.

## Branch results

### 1. Supervised DFWM residual embedding

The train-domain residual embedding was supervised with an 8-D descriptor of
actuator loss, damping, friction, delay, deadband, backlash, payload, and
observation noise. Weights 0.1, 1.0, and 5.0 produced nearly identical seed-7
results. DFWM K=5 multi-step RMSE remained about 0.5045 versus about 0.5038 at
K=0, so calibration slightly worsened performance.

Decision: **No-Go**. Supervising a free train-domain embedding does not teach
deployment probes how to infer the descriptor.

### 2. Supervised history encoder

The GRU history encoder was supervised first on pooled trajectories and then
on every trajectory independently. Weights 1 and 10 were tested. The final
per-trajectory version produced history-encoder K=5 RMSE 0.5984 versus
topology-only 0.4694 on seed 7.

Decision: **No-Go**. Stronger and correctly placed descriptor supervision did
not produce useful goal-directed Push predictions.

### 3. Residual-only Pivot

Residual-only was the only branch with a seed-7 K signal: 0.4054 at K=0 to
0.3788 at K=5, a 6.56% improvement. The expansion did not replicate:

| Seed | K=0 | K=5 | Relative change |
|---|---:|---:|---:|
| 7 | 0.4054 | 0.3788 | +6.56% |
| 17 | 0.4842 | 0.5109 | -5.52% |
| 27 | 0.3695 | 0.3723 | -0.75% |

Only 1/3 seeds improved, so the branch failed the direction gate.

## Overall decision

**G1 remains No-Go.** Neither factorized latent optimization, supervised
train-domain residual codes, supervised history inference, residual-only
adaptation, nor zero-shot structured dynamics provides a stable mechanism
signal under the corrected goal-directed Push task.

The earlier 15.8% result remains reproducible only for random excitation in a
Push scene. It must not be presented as few-shot recovery or goal-directed
Push evidence.

## Recommended project decision

Stop tuning the current DFWM implementation. The next step is a research-level
architecture change with a separately trained system-identification module and
an explicit residual dynamics correction, or a thesis/benchmark Pivot centered
on when topology and residual conditioning fail. Do not enter G2 or spend a
5-seed/10-seed budget on the current model.

## Artifacts

- `results/final/push_residual_supervision_seed7_sweep_20260819.csv`
- `results/final/push_history_supervision_seed7_sweep_20260819.csv`
- `results/final/push_residual_only_pivot_3seeds_20260819.csv`
