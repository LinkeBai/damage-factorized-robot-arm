# BT-DPWM Z76 Ablation and Uncertainty Audit

## Scope

These are post-confirmation diagnostic ablations. They reuse frozen checkpoints,
paired trajectories, and rollout targets. They do not change the Z76 decision
and are not a second opportunity to tune on seeds 57/67.

## Remove nested support memory

Replacing Z75 with the otherwise identical Z74 single-window rule produced no
negative BT-own gain on confirmation seeds. Seed57 K50 mean own gain was 12.84%
and seed67 remained at 0%. The acceptance timing changed on seed57 D2, but this
two-seed ablation did not reproduce a safety failure. Therefore the independent
confirmation set alone does not identify nested memory as necessary. Its direct
counterexample remains the development audit in which the pre-memory rule forgot
earlier support evidence and seed47 suffered negative transfer.

## Remove only the posterior standard-deviation threshold

The diagnostic config keeps the uncertain encoder, minimum fit count, unified
hysteresis, permanent z=0 fallback, and nested support validation, but omits the
hard `context_mean_std <= 0.30` acceptance test.

Across development seeds 7/17/27/37/47, all rollout rows were exactly identical
to Z75: the uncertainty threshold did not change one decision. On confirmation
seed57 it was also inactive. On seed67 it alone rejected a D3-unseen K25 context
with mean posterior standard deviation 0.464. Removing the threshold accepted
that context and reduced overall RMSE from 0.26596 to 0.22851, a 14.08% own
improvement, without negative gain at later budgets.

Across all seven evaluated seeds the no-threshold rule accepted 12 distinct
domain/budget updates. Eleven had mean standard deviation below 0.30; the sole
above-threshold update was beneficial. This is not enough data to prove that the
threshold should be removed, because accepted candidates are selected by support
validation and the sample is small. It does prove that the absolute 0.30 cutoff
is not empirically calibrated as a rollout-risk classifier: it was inactive on
development data and over-conservative on the only decision it changed.

## Consequence

The current narrow safety claim is carried by paired support validation,
hysteresis, and reversible z=0 fallback. Posterior spread should be reported as
a diagnostic until a development-only calibration experiment demonstrates risk
ranking or coverage. It must not be presented as a calibrated probability of
safe adaptation. Any replacement (for example, a held-out risk bound conditioned
on topology and budget) requires new development data and new untouched
confirmation seeds; seeds 57/67 cannot be reused to validate that change.
