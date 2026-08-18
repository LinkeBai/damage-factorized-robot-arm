# G1 Push formal gate (2026-08-18)

## Protocol

- Task implementation: MuJoCo arm plus movable block, 14-D state
- Methods: DFWM, topology-only, history encoder, parameter-matched,
  monolithic-matched, residual-only
- Seeds: 7, 17, 27, 42, 51
- Calibration shots: K = 0, 1, 2, 5
- Test domains: D2 and D3 with `mixed_composition`
- Training: 60 epochs per seed on the same generated trajectories
- Artifact: `results/final/push_6methods_5seeds_20260818.csv`

## K=5 aggregate

| Method | One-step RMSE | Multi-step RMSE |
|---|---:|---:|
| DFWM | 0.0349 | 0.1589 |
| topology-only | 0.0377 | 0.1888 |
| history encoder | 0.0354 | 0.1677 |
| parameter-matched | 0.0360 | 0.1631 |
| monolithic-matched | 0.0380 | 0.1831 |
| residual-only | 0.0344 | 0.1614 |

DFWM is 15.8% lower than topology-only on mean multi-step RMSE. It wins in
3/5 seeds, but the paired hierarchical bootstrap interval for the absolute
difference is `[-0.0049, 0.0731]`, which crosses zero.

## Mechanism check

DFWM multi-step RMSE by calibration amount:

| K | 0 | 1 | 2 | 5 |
|---|---:|---:|---:|---:|
| RMSE | 0.1573 | 0.1579 | 0.1585 | 0.1589 |

The residual latent adaptation does not improve DFWM as K increases. The K=5
advantage over topology-only is already present at K=0 and therefore does not
demonstrate the proposed few-shot residual-identification mechanism.

DFWM also does not significantly outperform history encoder,
parameter-matched, monolithic-matched, or residual-only on multi-step RMSE.

## Gate decision

**G1 No-Go for the current claim.** The numerical 15.8% comparison is
reproducible, but its proposed interpretation is not. The G1 requirements that
prediction/control generally improve with K and that factorization show a
stable advantage are not met.

Recommended pivot: treat the current result as robust zero-shot structured
dynamics evidence, or redesign the calibration objective/data so that K
contains identifiable residual information before rerunning G1.

## Remaining protocol caveat

The benchmark uses the Reach split metadata and random action excitation inside
the Push scene. It is a contact-dynamics prediction benchmark, not yet a
goal-directed Push control benchmark. A paper-level Push claim requires a
Push-specific immutable split, verified block displacement/contact coverage,
and saved calibration/evaluation trajectories.
