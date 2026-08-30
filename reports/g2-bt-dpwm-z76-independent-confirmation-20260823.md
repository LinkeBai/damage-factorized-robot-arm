# BT-DPWM Z76 Independent Confirmation

## Frozen protocol

Seeds 57 and 67 were fixed before either checkpoint existed. They use the
unchanged Z69 base mechanism, Z70 fair adapters, Z65 uncertainty encoder, and
Z75 nested-support safety gate. No seed was replaced and no threshold was
changed after observing confirmation results. The statistical unit is seed;
shared and BT are paired on the same domains, transition prefixes, and targets.

## Failure-to-recovery chain

The unmodified independent-expert V0 was not stable across confirmation seeds.
Seed57 improved free-arm rollout by 6.14% versus the parameter-matched shared
model but regressed object rollout by 2.82%. Seed67 regressed free-arm rollout
by 25.98% (32.28% versus the compute-matched shared model). Both runs therefore
remain V0 NO-GO evidence.

On seed67, Z32 shared-robot initialization reduced the free-arm regression to
16.06%. Z69, whose mechanism change is to zero untrained topology input columns
while retaining analytic projection, recovered free-arm to +0.47% and overall
to +0.27%. On seed57, Z32 already gave +40.61% free and +39.88% overall but
-29.39% object; Z69 retained positive free/overall (+2.21%/+2.05%) and reduced
the object regression to 20.42%. These Y0 models remain NO-GO because object
rollout is not yet uniformly competitive; they are dependencies of the safe
few-shot model, not selectively omitted failures.

## Independent Z75 result

| transitions | 0 | 5 | 10 | 25 | 50 |
|---:|---:|---:|---:|---:|---:|
| BT own gain (%) | 0.000 | 3.880 | 3.880 | 8.065 | 8.065 |
| shared own gain (%) | 0.000 | 3.933 | 3.933 | 6.440 | 8.686 |
| BT minus shared gain (pp) | 0.000 | -0.054 | -0.054 | +1.625 | -0.621 |
| BT relative shared (%) | +0.833 | +0.770 | +0.770 | +2.417 | +0.092 |

Every BT seed/domain/budget own gain is non-negative, the aggregate BT curve is
monotonic, and constraint violation is exactly zero. The K50 mean BT-own gain
is +8.07%; its two-seed bootstrap interval is +3.24% to +12.89%. Thus the
confirmation supports the narrow safety, reversibility, and useful-adaptation
claim.

The preregistered paired equivalence gate does **not** pass. At K25 the paired
BT-minus-shared bootstrap lower bound is -0.051 percentage points, inside the
1pp margin. At K50 it is -1.191pp, missing the -1pp margin by 0.191pp. The final
mean BT-relative-shared value is +0.092%, but the paired lower-bound criterion
takes precedence. This result does not support a sample-efficiency equivalence
or superiority claim on the independent confirmation set.

## Decision and next evidence

Z76 is a confirmation pass for own-baseline safety/monotonicity and a NO-GO for
the stronger paired sample-efficiency gate. The mechanism remains frozen. The
next G2 work is mechanism ablation, uncertainty calibration, robustness, and a
compute/failure ledger; these analyses must explain the K50 paired gap without
retuning Z75 on seeds 57/67. A new confirmation set would only be justified
after a separately motivated, development-only mechanism change.

Authoritative artifact:

- `runs/g2_bt_dpwm_z76_confirmation/two_seed_confirmation_v1/summary.json`
