# ICRA 2027 Evidence Contract (Frozen Before New Confirmation Runs)

## Purpose

This contract prevents score inflation and post-hoc experimental storytelling.
The current paper starts at **3.2/5 (5/10, weak reject)**. A score of 4+/5 is
permitted only if new evidence satisfies the frozen gates in
`config/experiment/icra_2027_evidence_contract_v1.yaml`.

## Decisive rule

Large revision progress is not an acceptance score. The paper cannot receive
4+/5 while its only task-level evaluation remains No-Go and it has no other
independent external-validity chain. Honest reporting of a failure improves the
limitations assessment but does not increase Evidence or Significance.

## Development/confirmation separation

- Seed 27 is development-only because it has already influenced method design.
- Seeds 37 and 47 are also not untouched: their results were inspected while the
  state-isolation wrapper and paper claim were designed.
- New training seeds 57/67/77/87/97 are reserved for confirmation.
- A new trajectory seed evaluated on an old checkpoint is not an independent
  training seed and must never be counted as one.
- After the first confirmation seed, no controller or model hyperparameter may
  change. Failure returns the method to development and requires a new future
  confirmation set.

## Closed-loop diagnosis already obtained

The existing seed-27 development run with 64 candidates and horizon 10 does not
support selective reranking. On `push_eval_00`, carrier-screen reaches 2.50 mm,
whereas selective-IPWM reranking reaches 11.63 mm. The result rejects immediate
multi-seed expansion of that controller. The next justified experiment is a
candidate-ranking calibration diagnostic: compare carrier and SI-IPWM predicted
cost differences with realized short-horizon cost differences on identical
candidate actions. Only a positive, pre-specified ranking signal can justify a
new controller gate.

## Required evidence before 4+/5

1. Five untouched training seeds, not merely new query trajectories.
2. At least three lock locations, four physics families, and ten targets per
   task condition.
3. Matched carrier/full-state/selective comparisons, a parameter-matched
   baseline, a strong robust/adaptive baseline, and mechanism ablations.
4. Hierarchical paired intervals over training seeds and within-seed targets.
5. Consistent task-level improvement or an equally strong independent robotics
   validation chain. Without hardware, simulation breadth must carry this load.
6. Raw rows, frozen configs, latency/cost measurements, failure ledger, and an
   end-to-end reproduction command.

## Current decision

**Evidence collection remains active.** Repackaging is not an allowed route to
the target score.
