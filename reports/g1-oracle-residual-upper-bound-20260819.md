# G1 oracle residual upper bound (2026-08-19)

## Question

Does the current DFWM improve when it receives the simulator's true continuous
residual parameters, bypassing deployment identification entirely?

## Protocol

- seed 7, 40 epochs
- corrected goal-directed Push task
- D2/D3 with `mixed_composition`
- residual embedding supervision weight 100
- K=0 DFWM versus the same DFWM with the true 8-D residual descriptor

## Result

| Method | One-step RMSE | Multi-step RMSE |
|---|---:|---:|
| topology-only | 0.0478 | 0.5135 |
| DFWM K=0 | **0.0465** | 0.4891 |
| DFWM oracle residual | 0.0469 | **0.4748** |

Oracle residual improves multi-step RMSE by 2.9% relative to DFWM K=0 and by
7.5% relative to topology-only. The oracle direction is consistent in D2 and
D3, but it is below the predeclared 10% upper-bound threshold.

## Decision

The world model can use residual information, but only weakly. Identification
is not the sole bottleneck: even perfect residual knowledge leaves a small
upper bound. Stop tuning latent/history inference for the current concatenated
context architecture.

The justified next architecture is an explicit residual dynamics correction:

`next_state = base_topology_dynamics(state, action) + residual_correction(state, action, residual)`

This separates nominal/topology prediction from continuous residual effects
instead of asking one recurrent context pathway to disentangle both.
