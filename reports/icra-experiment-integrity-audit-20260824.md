# ICRA Experiment Integrity and Sufficiency Audit (2026-08-24)

## Verdict

**No evidence of fabricated simulation numbers was found, but the current
evidence package is not yet strong enough for the claims as written or for a
competitive ICRA 2027 submission.** The 18 primary object-improvement values
are reproducible from the three tracked aggregate metric files and match
`matched_gate_summary_v2.json`. However, one stated non-regression claim is
false, raw per-rollout evidence and uncertainty intervals are absent, model
checkpoints are untracked, and real-arm task evidence is still missing.

Recommended status: **G2 simulation mechanism demonstrated; G2 evidence freeze
must be reopened for an audit-only evaluation export; G3 real-arm validation is
not complete. Do not present the current manuscript as submission-ready.**

## Scope and sources

Audited sources:

- `paper/main.tex` and `paper/main_zh.tex`;
- `scripts/summarize_g2_r0_physical_context_gate.py`;
- `scripts/evaluate_g2_r0_core_metrics.py`;
- the three `k25_matched_adapter_d3_metrics.json` files for seeds 7/17/27;
- matched control, K0, no-depth-risk, and component-ablation JSON files;
- frozen seed-specific YAML files and Z65/Z69/Z70 dependencies;
- Git tracking and chronology;
- focused model/audit tests.

The official ICRA 2027 call requires an eight-page complete, double-anonymous
paper. The official reviewer guidance grades the paper as a whole; technical
errors can justify summary rejection, while lack of real experiments alone is
not a summary-rejection condition. For this manipulation claim, however,
real-arm evidence is practically important to technical credibility.

Official references:

- https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
- https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/
- https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-associate-editors/

## Result-by-result findings

### Primary object table: internally consistent (PASS with limitations)

- All 18 paper values exactly match the tracked frozen summary after rounding.
- Re-running the summarizer produces no Git diff.
- Range: 2.0430085882% to 37.8357085024%.
- Seed27 was committed as confirmation before the final synthesis commit, which
  supports but does not cryptographically prove the “untouched” statement.
- The three seeds use the same final architecture and hyperparameters. Seed7 is
  explicitly a regression run after seed17 selection; only seed27 is an
  independent confirmation seed.

### Analytic feasibility: supported (PASS)

- All evaluated locked-coordinate violation values are exactly zero.
- Projection is explicit in the evaluator, rather than learned through a loss.
- Focused tests covering projection/context behavior pass (38/38).

### Free-joint non-regression: contradicted (FAIL)

The manuscript says free-joint changes remain within 5%. This is false for:

- seed7, D3 mixed-unseen, H50: **-5.9652% free-joint improvement**, i.e.
  5.9652% regression.

The frozen YAML gate is `maximum_free_arm_regression_pct: 5.0`, but the summary
script never evaluates that gate. The manuscript and final report therefore
overstate the evidence. Two H50 cells also have negative overall improvement:
seed7 mixed-unseen (-5.7511%) and seed17 composition (-0.5676%). These do not
invalidate the object-specific claim, but must be disclosed.

### Pusher/control statement: only partially supported (QUALIFIED)

- Six of 18 primary D3 cells have negative relative pusher improvement.
- Control JSON supports the reported small absolute IID differences, but the
  paper should not imply universal pusher improvement.
- The defensible statement is task-object improvement with small measured
  task-space collateral changes, not full-state Pareto dominance.

### K25 mechanism and ablations: numerically traceable (PASS with scope limit)

- The seed17 K0/no-depth/full-K25 values are present in tracked JSON artifacts.
- Geometry, latent, intervention, and depth-risk ablations match the reported
  values.
- These ablations are only on seed17. They establish a mechanism case study,
  not population-level component necessity.
- K is explicitly non-monotonic; only the frozen K25 policy is supported.

## Statistical and reproducibility audit

### Evaluation sample size is small at long horizons

The evaluator uses three trajectories per test domain, each 150 steps, and
non-overlapping terminal windows. Effective terminal-error samples per
seed/domain/method are therefore approximately:

| Horizon | Samples |
|---:|---:|
| H10 | 45 |
| H25 | 18 |
| H50 | 9 |

The current JSON retains only aggregate RMSE. It does not retain per-trajectory
or per-window errors, so paired bootstrap intervals, dispersion, outlier
inspection, and independent recomputation from raw rows are impossible from
the committed evidence package. The 18 cells are also correlated across
horizons/domains and must not be treated as 18 independent replications.

### Artifact provenance is incomplete

- Aggregate JSON, configs, scripts, and reports are tracked.
- The three final `model.pt` files, shared checkpoints, context encoders, and
  matched adapters are ignored by Git and have no committed manifest mapping
  them to SHA-256 hashes.
- Locally observed final-model hashes are:
  - seed7: `f757a3952e98d4af5f72248721f5641aa94748a14f86e3180be6d8ef26e5162e`
  - seed17: `afb59f6b3d4e41566e6396fcc7f7ae1d4e50dcd1bdb8b8d23f66a29e3e37e461`
  - seed27: `af1a5c0e2d9cef67b476b6259bdcad9ecd010b1ad16b92a223313e6e72238e58`
- Exact command lines, environment lock, GPU/software versions, cache hashes,
  and raw evaluation row hashes are not bundled into one immutable manifest.

Thus a local rerun is possible while these files remain on this machine, but a
fresh clone cannot reproduce the paper.

## ICRA sufficiency assessment

| Dimension | Current level | Submission need |
|---|---|---|
| Novel method | Plausible, narrowed claim | sharpen against closest methods |
| Primary simulation effect | Directionally positive | add paired uncertainty and more evaluation trajectories |
| Independent confirmation | One seed (27) | add at least 2 untouched confirmations if time permits |
| Strong baselines | One main matched shared baseline | add same-protocol damage-conditioned/factorized baselines |
| Safety/constraints | Analytic zero violation | correct the failed 5% free-joint claim |
| Mechanism ablation | One development seed | replicate decisive ablations on confirmation seed(s) |
| Real manipulation | Protocol only | collect D2/D3/intact pushing results and video |
| Reproducibility | Partial | raw rows + hashes + immutable manifest + environment |
| Paper compliance | IEEE double column, under 8 pages | retain double anonymity and final 8-page cap |

Estimated reviewer posture today: **C / low-borderline to reject**, mainly due
to missing real task evidence, weak statistical reporting, one contradicted
safety/non-regression statement, and incomplete fresh-clone reproducibility.
This is not a judgment that the method is invalid; it is a judgment that the
current evidence does not yet carry the strength of the manuscript narrative.

## Mandatory actions before submission

### P0 — integrity corrections (do immediately, no retraining)

1. Change “free-joint within 5%” to report the actual worst regression, or
   change the formal gate and clearly disclose the exception.
2. Remove the claim that paired rollout rows back every aggregate until those
   rows are exported and committed.
3. Label the result as object-specific selective improvement, not overall
   world-model dominance.
4. Add an immutable manifest with checkpoint/config/cache hashes and commands.

### P0 — audit-only reevaluation (reuse checkpoints; no retraining)

1. Extend the evaluator to export per-trajectory/per-window squared errors.
2. Re-evaluate the frozen seed7/17/27 checkpoints with at least 20--30 test
   trajectories per domain, keeping identical actions for both methods.
3. Report paired bootstrap 95% intervals and absolute RMSE, not only percent
   improvement. Treat seed, not horizon cell, as the replication unit for broad
   claims.
4. Run the decisive no-intervention/no-geometry/no-latent/no-depth ablations on
   seed27 using the frozen checkpoints.

### P0 — real-arm evidence

Collect intact/D2/D3 pushing with frozen K25 and matched shared baseline,
report every repetition and abort, and include the setup/calibration figure and
short submission video. Current G0 lock-hold measurements are safety readiness,
not task validation.

### P1 — stronger comparative evidence

Under the same data, projection, K25 and optimization budget, include at least:

- shared + analytic projection + matched adapter (already present);
- damage-conditioned monolithic model;
- factorized/dual-expert predecessor or no-intervention model;
- the proposed complete model.

## Reusable versus rerun

Reusable without retraining: all three frozen model checkpoints, Z65 context
encoders, Z70 matched adapters, MuJoCo trajectory-generation code, configs,
component weights, K25 rule, analytic projection tests, and existing development
history. Required work is primarily evaluation/export plus real-arm collection.
Retraining is necessary only if the corrected free-joint gate is kept at a hard
5% requirement and the frozen model must satisfy it rather than disclose the
single exception.

