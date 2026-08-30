# SI-IPWM CCFA / ICRA Re-review — Corrected Assessment

- Review date: 2026-08-28
- Manuscript: `output/pdf/main.pdf` / `paper/main.tex`
- Target: IEEE ICRA 2027 contributed paper
- Review mode: full scientific and evidence review, with version progress reported separately
- **Corrected five-point assessment: 3.2/5**
- **Calibrated overall: 5/10 (borderline negative / weak reject)**
- Scholarly confidence: 4/5

> This report supersedes the earlier 4.0/5 assessment. That assessment incorrectly
> fused substantial revision progress with absolute ICRA readiness and under-penalized
> the negative closed-loop result, lack of hardware evidence, and narrow evaluation.

## Decision summary

The manuscript is now clear, technically honest, and substantially stronger than its
earlier version. Its central state-isolation invariant is valid and the prediction
results are internally consistent. However, the current evidence supports only a
narrow simulation claim: under one planar-pushing environment and one held-out lock
family, SI-IPWM preserves the object prediction of the full-state intervention while
preventing collateral free-joint changes.

The only task-level closed-loop evaluation is a **No-Go**: SI-IPWM is better than the
carrier in only one of three reported checkpoints and is worse in the other two.
That result improves the paper's honesty, but not its demonstrated robotics impact.
Because there is also no real-robot experiment, no broad multi-task simulation suite,
and no reproduced closest-method comparison, the manuscript lacks an independent
evidence chain showing that the prediction improvement produces useful manipulation
behavior. This is a major Evidence/Significance limitation and caps the paper below
4/5 at ICRA.

## Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 4 | Secs. III–IV: private physical rollouts plus carrier-invariant published robot state; Proposition 1 | Mechanism is clean but structurally simple and closest adaptive world-model systems are not tested. Raise with a sharper theoretical delta and matched contemporary baselines. |
| Soundness | 4 | 4 | Proposition 1; full-state versus isolated audit; exact carrier equality | The invariant is valid for the published robot block, but it is not a task-success or physical-correctness guarantee. Raise only with stronger scope analysis across interventions. |
| Evidence | 3 | 5 | Main prediction tables; 27/27 post-design query cells; guarded-MPC table | Three training checkpoints, one simulator/task/object/lock family, and closed-loop No-Go. Raise to 4 with independent training seeds plus decisive task-level validation across failures, or credible hardware evidence. |
| Significance | 2 | 4 | Abstract, controller audit, limitations | The paper does not demonstrate that safer state publication improves manipulation outcome; simulation prediction gains alone are narrow. Raise with consistent control/task benefit or compelling broader robotics consequences. |
| Clarity | 4 | 5 | Six-page rendered manuscript; claim gates and failure ledger | Clear and unusually honest, but dense tables constrain accessibility. |
| Reproducibility | 4 | 4 | Frozen configs, checkpoint/query seeds, raw rows, scripts, unit tests | Good artifact trail; raise with a clean end-to-end reproduction command and independently verified package. |
| Ethics / Limitations | 5 | 5 | Simulation-only statement, invalid-metric rejection, explicit No-Go, corrected camera description | Clear strength; no deduction. |

**Overall:** 5/10 | **Five-point summary requested by author:** 3.2/5

**Recommendation:** weak reject / borderline negative

**Verdict:** A positive, repeated task-level result would move the assessment toward
6–7/10. Hardware is not formally mandatory, but without hardware the simulation
evidence must be substantially broader and more decisive than it is here.

## Why the No-Go matters

The guarded-MPC result is not merely a benign negative result. The paper motivates a
robotics intervention world model, so task-level utility is a central significance
test. Terminal distances are:

- checkpoint 27: carrier 13.75 mm, SI-IPWM 19.45 mm;
- checkpoint 37: carrier 18.77 mm, SI-IPWM 15.67 mm;
- checkpoint 47: carrier 34.25 mm, SI-IPWM 39.11 mm.

Thus SI-IPWM wins in only one of three checkpoints. This does not invalidate the
state-isolation proposition or object-prediction result, but it prevents a claim that
the method improves control and substantially weakens the case that the contribution
matters for manipulation performance.

## How to interpret the absence of real-robot experiments

ICRA's official submission rules do not state that every paper must include a physical
robot experiment. A simulation-only paper can be competitive when its algorithmic or
theoretical contribution is strong and its simulation evaluation is substantial,
diverse, and decisive. Here, however, the contribution is an applied world-model
intervention for damaged manipulation, while both external-validity routes are weak:

1. no hardware or sim-to-real evidence; and
2. no positive closed-loop simulation result.

Therefore the lack of hardware is not a desk-reject condition by itself, but it becomes
a major cumulative weakness in this particular evidence package.

## Relative progress versus absolute readiness

### Relative progress

The revision is materially improved: it removes unfilled hardware claims, corrects the
camera architecture, isolates the free-joint failure, adds a formal invariant and
matched ablation, adds post-design query trajectories, rejects the invalid 50 mm
success metric, and reports the controller failure. This is genuine progress.

### Absolute ICRA readiness

Progress does not equal acceptance readiness. Under the current-paper-only assessment,
the blocking evidence remains: negative closed-loop outcome, no hardware, only three
training checkpoints, query trajectories rather than new training initializations,
one task/object/failure setting, and no matched reproduction of closest systems.

## Desk checks

- Length/template/topic: pass for an eight-page-complete-paper ICRA 2027 submission;
  the manuscript is six pages and within robotics scope.
- Minimum scientific quality: pass.
- Reviewability and limitation disclosure: pass.
- Anonymity/compliance: must be rechecked on the final submission package.
- Prompt-injection/hidden-review manipulation: none observed in the reviewed manuscript.
- Desk-rejection risk: low on format, but substantive rejection risk is medium-high.

## Score-change conditions

| Change | Condition | Likely dimensions | Expected movement |
|---|---|---|---|
| Raise | Pre-registered closed-loop protocol succeeds consistently over enough independent training seeds and multiple lock/failure conditions | Evidence, Significance | +1 to +2 overall |
| Raise | If no hardware is possible, add broad simulation tasks/objects/failures, matched strong baselines, uncertainty, latency, and statistical intervals | Novelty, Evidence, Significance | +1 overall if decisive |
| Raise | Add even a modest but credible external-camera real-robot validation with repeated trials and honest failure statistics | Evidence, Significance | +1 overall, depending on result |
| Lower | The 27/27 query result does not reproduce across new training initializations | Soundness, Evidence | -1 or more |
| No quick change | Merely rewording the paper or presenting No-Go as “realism” | Evidence, Significance | no movement |

## AC-style synthesis

- Agreement: clear manuscript, valid isolation invariant, strong limitation honesty.
- Decisive positive axis: exact state-isolation property with auditable prediction gains.
- Decisive negative axis: no demonstrated task-level benefit and no physical validation.
- AC stance: weak reject; encourage resubmission after new evidence rather than further
  rhetorical packaging.
