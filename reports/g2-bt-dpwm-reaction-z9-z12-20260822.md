# BT-DPWM reaction attribution: Z9--Z12

## Fixed constraints

The h136/240 shared baseline, three seeds, four test domains, parameter ceiling,
damage projection and rollout horizon remain unchanged. Seed 7 is development;
seeds 17/27 are never used to choose a hyperparameter.

## Z9: global magnitude

On seed 7, frozen Z5 peaks at scale 0.75 with overall +5.12%. Locked replication
gives three-seed means free +0.06%, object +21.99%, overall +0.99%, with 4/12
regressions. Scale zero gives overall +0.75%. Magnitude alone is insufficient.

## Z10: physical-coordinate reaction

The adapter input is changed from arbitrary robot hidden coordinates to ten
per-joint physical features. An 80-unit bottleneck uses about 1,042 adapter
parameters, below Z5's 1,146. Seed 7 at fixed deployment scale 0.30 reaches free
+3.94%, object +25.60%, overall +5.25%. Locked three-seed audit reaches free
+0.86%, object +21.97%, overall +1.76%, with 4/12 regressions. Seed 7's
mixed-unseen cell remains overall -6.79%, identifying held-out physics as a
remaining failure.

## Z11: group-robust training

A smooth worst-domain objective across the 12 training domains reaches seed-7
overall +5.31% at scale 0.30, but mixed-unseen only changes from -6.79% to
-6.35%. The naive per-domain batching also costs about 12x wall time. No
cross-seed expansion is justified.

## Z12: bounded event memory

A parameter-free analytic contact trace retains reaction after contact and
decays geometrically. At threshold +5 mm and decay 0.95, seed 7 reaches free
+3.41%, object +25.65%, overall +4.72%, with 0/4 regressions. It improves safety
relative to ungated reaction but misses the +5% gate.

All Z9--Z12 candidates are NO-GO under the full objective. The next mechanism
must constrain reaction under held-out physics using deployable information and
must not select per-seed scales or checkpoints from test data.
