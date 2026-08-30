# BT-DPWM Fixed-Budget Asymmetric Gates Z4--Z5

Date: 2026-08-22. Strong baseline: shared h136/240, 338,102 parameters.

## Z4a: random asymmetric allocation

Robot128/object56 uses 337,518 parameters. On seed 7 primary D3 it improved
object by 43.81% but regressed free-arm by 90.13% and overall by 80.30%.
Simple width reallocation with joint-only robot identification is NO-GO.

## Z4b: scaffold-specialize BT-DPWM

Robot136 is initialized from the strong shared model and frozen; its shared object
head is discarded. A new independent object32 recurrent graph is then trained on
the fixed scaffold. Total parameters are 336,910. Seed 7 primary D3 improved
free-arm 2.83%, object 42.13%, and overall 6.35%. Across its four test domains,
means were +0.65%, +25.57%, and +2.12%, with one regression. The strict +5%
cross-domain goal is not met, but both blocks improve on average.

## Z5: fixed-budget reaction adapter

A zero-initialized rank-8 robot reaction adapter (1,146 parameters) was added to
Z4b. The complete model has 338,056 parameters, 46 fewer than the baseline. Only
the adapter is updated for 40 epochs after object training.

Seed 7 primary D3 passed strongly (+5.30% free, +42.17% object, +8.64% overall),
and its four-domain mean overall was +5.03%. The result did not replicate:
seed 17 primary overall was +1.23%, while seed 27 was -8.81%. Across all three
seeds and four domains, means were -0.36% free, +22.00% object, and +0.59%
overall, with 4/12 overall regressions. Z5 is formally NO-GO.

## Decision

The fixed-budget scaffold-specialize mechanism robustly improves object rollout,
but an unconstrained train-loss-selected reaction adapter does not generalize across
seeds. The next permitted change is validation-selected reaction training under the
same architecture and parameter budget, including the zero-reaction checkpoint as
a valid choice. No new model family or parameter increase is authorized.
