# Strict primary-arm evidence-contract audit (2026-08-31)

## Decision

The authoritative Git/GitHub repository has been recovered and synchronized.
The original IPWM modules are complete; the earlier missing-source finding was
specific to a separate unversioned workspace copy.

The frozen primary-arm configuration already names five methods and three
decisive ablations, but it is currently a specification rather than an
executable unified experiment. The machine-readable audit is
`results/final/primary-evidence-contract-audit.json`.

## Current implementation coverage

The authoritative SI-IPWM implementation contains analytic topology projection,
a mechanism-matched carrier, private intervention rollout, and selective object
publication. Paired counterfactual action-ranking is now implemented without an
external ranker. All eight frozen mechanism cells now have executable
implementations. No cell yet has a formal same-protocol result. Therefore:

- historical selective-IPWM prediction results cannot be relabelled as
  decision-focused results;
- the separate 128-candidate sequence experiment cannot be relabelled as the
  exact SI-IPWM because that compact model predicts all non-locked coordinates;
- Z82--Z85 structural ablations use another BT-DPWM protocol and cannot fill the
  primary-arm unified table;
- all eight rows must be executed through one new driver sharing data, candidate
  sequences, model capacity rules, planning budget, seeds, and statistics.

## First executable decision-focused smoke

The paired soft-regret objective is now implemented directly on the terminal
object state predicted by the existing world model. It adds no parameters or
external ranker. A loader enforces identical initial state, goal and diagnosis
within every candidate group and programmatically permits only D2/D4 for
training and validation.

The first original-arm smoke used four D2/D4 groups, 32 candidates per group,
50-step actions and two epochs. The loss produced a finite nonzero gradient
(`0.505`), but validation selected epoch 0. Carrier, full-state and selective
publication consequently tied, with Spearman `-0.0362`, Kendall `-0.0590` and
top-1 regret `0.01238`; the candidate oracle cost was `0.00446`. This is a
pipeline pass and performance No-Go, not paper evidence. It shows substantial
candidate headroom but no learned selection at the selected checkpoint.

The same-checkpoint evaluator now reports shared baseline, mechanism-matched
carrier, full-state intervention and selective publication separately. The
first implementation attempt incorrectly used the shared model as carrier and
was discarded; the corrected carrier is a deep copy of the candidate with only
intervention object heads zeroed.

## Frozen implementation order

1. Complete the unified driver so every row shares data, candidate actions,
   optimization budget and checkpoint-selection rules.
2. Execute projection, path-support and paired-loss ablations only through that
   unified driver.
3. Select hyperparameters on D2/D4 development only; D3 remains confirmation.
4. Run a single development seed before expanding to 7/17/27; do not read or
   retune on D3 confirmation outcomes.

## No-projection ablation wiring check

The analytic projection can now be disabled while preserving the exact model
architecture and weights. Unit tests demonstrate that locked position and
velocity can drift when this switch is disabled; 46 focused tests pass. A tiny
D2/D4 smoke (four groups, 32 candidates and two epochs) completed with overall
improvement `-0.14%`, Spearman `-0.0415` and top-1 regret `0.01238`. The sampled
rollout happened to report zero lock violation, so this run is only a wiring
check and a performance No-Go. A purpose-built lock-stress set and formal
multi-seed run remain necessary.

## Global matched-capacity wiring check

The global comparator reads the same 12-dimensional geometry/contact input as
the selective correction but may publish to every one of the 14 state
coordinates. Its frozen rank-10 head has 284 parameters versus 276 for the
rank-16 selective head, an eight-parameter difference over the whole model;
hard projection remains active. A four-group, 32-candidate D2/D4 smoke selected
epoch 0 and produced object improvement `-2.55%`, overall improvement `+6.55%`,
Spearman `-0.0362` and top-1 regret `0.01238`. This proves the comparator is
executable and not parameter-starved, but is a performance No-Go and not a
formal same-protocol result.

## Real-robot interface

`data/real_robot/push_trials_template.csv` and
`scripts/analyze_real_robot_push.py` define paired carrier/SI-IPWM logging.
Aborted trials remain in the ledger and are excluded, not imputed. Fewer than
10 complete pairs are automatically labelled pilot evidence.
