# IPWM Fault-Hybrid Counterfactual Gate (2026-08-29)

## Frozen hypothesis

At H10 under a diagnosed held-out joint lock, explicitly predicting contact
survival and using contact-conditioned response experts should improve both
cumulative object prediction and candidate-action ordering across GenkiArm and
Panda. Analytic lock projection and SI-IPWM state isolation remain the safety
boundary; this Gate tests only the missing contact/control response mechanism.

## Fair comparison

- 2,880 H10 branches: two actual MuJoCo robot structures, 80 current-contact
  prefixes per robot, six candidate actions, and three locks.
- GenkiArm j3 and Panda joint4 are held out; split is grouped by robot/prefix.
- Candidate: contact-survival mixture with two conditional response experts.
- Strong baseline 1: flat multi-task model receiving the same contact label.
- Strong baseline 2: flat non-mixture predictor on the same inputs.
- Parameter counts: 13,555 / 13,610 / 13,509; maximum mismatch 0.41%.
- Robot identity, solver force, future state, future contact as input, and target
  labels as input are forbidden.

## Results against the strongest baseline

| Seed | RMSE improvement | Both robots | Spearman improvement | Lower regret | Balanced accuracy | Brier | Gate |
|---:|---:|:---:|---:|:---:|---:|---:|:---:|
| 7 | -5.21% | No | +0.0179 | No | 0.613 | 0.346 | Fail |
| 17 | +4.54% | Yes | -0.0286 | No | 0.659 | 0.345 | Fail |
| 27 | +12.93% | Yes | +0.0223 | Yes | 0.633 | 0.383 | Fail |

The preregistered requirements are at least +10% RMSE, +0.10 Spearman, lower
regret, improvement on both robots, balanced accuracy at least 0.70, and Brier
at most 0.20 in at least 2/3 seeds. The candidate passes **0/3**.

## Decision

**No-Go.** Explicit mode factorization sometimes improves cumulative response,
but contact survival is poorly calibrated and the improvement does not reliably
translate to action ordering. Seed 27 is not a partial Go: it misses both mode
criteria and the ranking-effect threshold.

The frozen rule prohibits tuning mode threshold/count, H10, loss weight, head
count, hidden size, features, or seeds. The mechanism cannot be presented as a
paper contribution and does not authorize dual-task five-seed confirmation.

## Consequence

The repository now contains independent failures of:

1. a learned solver/constraint response operator;
2. graph-based cross-arm object/contact propagation;
3. an instantaneous analytic action-effect transfer operator; and
4. an explicit H10 contact-survival mixture.

Together these results reject the current family of “structured residual plus
one more propagation/mode component” solutions. A new proposal must introduce
new information or a formal principle, not another head. In particular, the
present H10 dataset contains no pre-action active-probe history spanning varied
contact/compliance physics, so it cannot test the original few-shot context-
identification premise. Building such a dataset is a new experimental program,
not an allowed post-hoc repair of this Gate.
