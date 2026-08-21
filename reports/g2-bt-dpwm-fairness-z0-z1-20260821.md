# BT-DPWM Fairness Audits Z0--Z1

Date: 2026-08-21. Frozen BT-DPWM Y6 checkpoints, seeds 7/17/27, four held-out
test domains, rollout depth 10.

## Z0: cross-domain audit against shared h96/120

Across 12 seed-domain cells, mean improvements were +6.20% free-arm, +2.21%
object, and +6.15% overall. Ten of 12 cells improved overall. The two regressions
were both small and confined to `D3__mixed_unseen` (seed 7: -0.65%; seed 17:
-0.80%; seed 27 improved +2.58%). The preregistered gate allowed at most one
regression, so Z0 is formally NO-GO despite positive means.

Deployment-cost snapshot on CUDA, batch 64, 10-step rollout:

| Model | Parameters | Rollout latency |
|---|---:|---:|
| shared h96 | 169,542 | 11.60 ms |
| shared h136 | 338,102 | 11.51 ms |
| BT-DPWM h96 | 319,494 | 19.84 ms |

BT-DPWM is parameter matched to shared h136 (5.5% fewer parameters), but its two
sequential recurrent blocks make inference about 1.72x slower in this benchmark.

## Z1: strong parameter- and epoch-matched shared baseline

To remove the remaining capacity/training-budget confound, shared h136 was
trained from scratch for 240 epochs at each seed on the identical cached data.

| Seed | Free-arm vs h136/240 | Object vs h136/240 | Overall vs h136/240 | Gate |
|---:|---:|---:|---:|---|
| 7 | -46.49% | +43.00% | -42.30% | NO-GO |
| 17 | -30.67% | +31.44% | -28.81% | NO-GO |
| 27 | -39.91% | +34.74% | -37.48% | NO-GO |
| **Mean** | **-39.02%** | **+36.40%** | **-36.19%** | **0/3 PASS** |

The sign pattern is consistent: BT-DPWM's independently identified object block
is substantially better, while its robot block is under-capacity or under-optimized
relative to the strong shared model. Overall RMSE is dominated by the ten robot
coordinates, so current Y6 is not a fair-baseline overall winner.

## Decision

Y6 remains evidence that block-coordinate object identification solves object
rollout, but its earlier overall PASS against h96/120 must not be presented as a
general method win. The next permitted intervention stays inside BT-DPWM: reallocate
the fixed ~338k parameter budget from the over-performing object block to the robot
block, then rerun the same Z1 gate. No new framework or naming is authorized.

## Y7: robot-update-budget attribution

Before changing capacity, Y7 tested the cheaper alternative explanation that the
robot block was merely under-trained. With architecture and seed fixed, robot
training was increased from 120 to 240 epochs at horizon 10; object training stayed
at 120 epochs/horizon 5. Robot training loss fell from 0.02291 at epoch 120 to
0.01108 at epoch 240. Nevertheless, on the primary D3 domain versus h136/240,
free-arm regressed 72.94%, object improved 46.57%, and overall regressed 64.09%.

Y7 is a seed-7 attribution NO-GO and is stopped without spending seeds 17/27. More
robot updates reduce the training objective but worsen held-out long rollout, so
simple update budget is not the cause. Capacity allocation also remains unproven.
The next diagnosis must distinguish robot representation/conditioning bias from
missing contact-related auxiliary supervision; it must not add epochs or rename the
method.
