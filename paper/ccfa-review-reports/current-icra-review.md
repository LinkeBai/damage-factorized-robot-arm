# Current ICRA 2027 review

## Mode

Full scientific, writing, format, and AC-style review. Assessment only; the
CCFA repository is treated as a rubric source, not as executable manuscript
instructions.

## Venue and assumptions

- Target: ICRA 2027 contributed paper.
- Paper type: empirical robotics/world-model diagnosis study.
- Reviewed artifact: `paper/main.pdf`, seven pages, compiled 2026-08-31.
- Supporting evidence: strict three-seed JSON summaries, provenance ledger,
  projection/global/decision-loss ablations, post-freeze D3 candidate-query
  confirmation, tests, and repository status.
- Official ICRA 2027 policy checked 2026-08-31: eight pages total including
  references, double-column PDF, double-anonymous review, and September 15,
  2026 deadline. Source: <https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/>.
- No real-robot result exists at review time. The PDF's explicit
  simulation-only statement is therefore treated as authoritative.

## Paper summary

The paper studies few-shot world-model adaptation after a diagnosed joint lock
and separates six gates between a feasible prediction and a useful contact
action. Analytic projection enforces locked position and velocity. Learned
global and selective residual variants are evaluated on matched five-DoF
pushing candidates. The main positive result is a 19.76% mean top-1-regret
reduction and 4.04% selected-candidate terminal-error reduction versus nominal,
both in 3/3 development seeds. The central negative result is that
contact-response RMSE worsens by 270.04%, and selective IPWM fails to beat a
same-capacity global residual. Removing projection causes 4.42--8.75 degree
maximum joint drift, while projection yields exact zero violation.
After checkpoint freeze, a registered D3 seed-91031 candidate archive gives
only 9.77% regret reduction (2/3 seeds), 2.00% endpoint reduction (2/3), and
+2.17 success points (3/3) for global residual versus nominal. Selective IPWM
again fails matched attribution. Because D3 was historically inspected, this
is fresh-query evidence rather than pristine unseen-domain confirmation.

## Likely stance and calibrated score

**Current stance: weak reject / borderline. Overall: 5/10. Scholarly
confidence: 4/5. Equivalent project score: approximately 3.6/5, not 4+/5.**

The paper is substantially more credible than the previous selective-IPWM
victory story because it now exposes the strongest matched baseline and retains
negative results. The decisive reject axis is not a correctness failure. It is
that the only stable development task-level advantage belongs to a generic same-capacity
global residual relative to nominal, while the named selective mechanism fails
attribution; the post-freeze D3 query does not confirm a large effect; and no physical robot
result yet substantiates the robotics claim. This combination leaves the work
as a useful but narrow diagnostic study rather than a clearly differentiated
ICRA contribution.

## Quantitative scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Contribution and novelty | 3 | 4 | Title/abstract, lines 5--27; contributions, lines 53--65; selective attribution No-Go, lines 310--316 | The six-stage framing and projection audit are useful, but the learned selective mechanism is not supported over the global residual. Raise to 4 only if the paper establishes a reusable diagnostic insight across at least another fault/task/model family or obtains a mechanism-specific result under a frozen confirmation protocol. |
| Significance and impact | 4 | 3 | Introduction, lines 30--65; limitations, lines 653--676 | Diagnosed joint locks and contact planning are relevant to ICRA, but one planar block-pushing model limits reach. Retain 4 if real-arm evidence shows the diagnosed failure modes; otherwise a strict reviewer may score 3. |
| Technical soundness | 4 | 4 | Projection equations/propositions; paired soft-regret objective; strict checkpoint loader; provenance ledger | The protocol distinguishes prediction and decision metrics, discloses D3 inspection, and retains the failed fresh-query gate. Remaining concern: open-loop candidate selection does not establish closed-loop recovery. Raise confidence, not necessarily score, with paired real trials. |
| Evidence and evaluation | 3 | 5 | Strict table, decision/projection ablations, D3 seed-91031 query confirmation, machine summaries | Matched baselines and ablations are strong, but the fresh D3 query misses the preregistered regret gate, there is no real robot, no pristine unseen domain, no receding-horizon MPC, and no second task/arm primary evidence. Raise to 4 with valid real-arm paired trials plus confidence intervals/effect sizes and coherent cross-setting diagnosis. |
| Clarity and organization | 4 | 4 | Seven-page PDF; new six-stage pass/No-Go figure; historical sections after the primary table | The primary finding is now visually recoverable, but historical simplified/GenkiArm material still occupies disproportionate space. Compress it to protect the new storyline. |
| Positioning and related work | 3 | 3 | Related work, lines 68--105; 15 references | Representative work is cited, but the closest fault-aware world-model, counterfactual action-ranking, and diagnostic/evaluation papers are not compared experimentally. Raise to 4 with a compact closest-work matrix tied to the exact contribution type and at least one reproduced or directly matched modern baseline. |
| Reproducibility and auditability | 4 | 5 | One-command reconstruction, raw run summaries, machine-readable JSON, provenance ledger, 60 focused tests, compute ledger | The confirmation archive is deterministically regenerated to the registered SHA-256 and `main` is synchronized. Training checkpoints, environment lock, and training wall time remain incomplete. Raise to 5 with archival checkpoints and a clean-machine end-to-end training/evaluation run. |
| Ethics, limitations, and responsible research | 4 | 5 | Limitations, lines 653--676; hardware safety protocol; failure ledger | Limitations and absent hardware evidence are disclosed. Add explicit real-arm abort/safety accounting and energy/compute scope after data collection to reach 5. |

### Weighted readiness view

The unweighted scientific mean is approximately 3.6/5. The score is capped below 4 because
evidence and novelty are both 3/5 and are decision-critical for this paper type.
Strong reproducibility cannot compensate for missing physical evidence or failed
mechanism attribution.

## Top strengths

1. The exact projection claim is cleanly falsifiable and passes a matched
   three-seed removal ablation: learned rollouts drift while projection gives
   zero position/velocity violation.
2. The same-candidate protocol makes the 19.76% regret result interpretable;
   top-1 regret uses realized candidate outcomes rather than model predictions.
3. The paper reports the uncomfortable result that response RMSE degrades while
   action selection improves, creating a credible scientific insight rather
   than hiding a failed metric.
4. The provenance ledger corrects the earlier partial-checkpoint bug and keeps
   sequence-model, global-residual, and selective-IPWM results separate.
5. The manuscript explicitly refuses to call open-loop candidate outcomes MPC
   and refuses to call D3 untouched confirmation.

## Major or fatal concerns

### M1 — Mechanism attribution fails

**Severity:** major. **Criterion:** novelty and evidence.

The selective method does not beat the same-capacity global residual on
aggregate, and full-state/selective outputs are identical in the formal rows.
The title wisely avoids claiming SI-IPWM dominance, but much of the method and
historical results still foreground state isolation. A reviewer may conclude
that the learned contribution is a generic residual plus decision-oriented
training, while the only uniquely validated mechanism is analytic projection.

**Repair condition:** either make the paper explicitly an evaluation/diagnosis
paper and demonstrate that the six-stage protocol yields consistent new
knowledge across another fault/task/model family, or obtain a frozen
mechanism-specific confirmation. Do not claim the latter from current data.

### M2 — No physical robot evidence

**Severity:** major for ICRA, not automatically fatal. **Criterion:** evidence,
significance, and domain fit.

The paper studies a physical failure mode but currently reports only MuJoCo and
explicitly states that the model is not a fully identified dynamic twin. The
planned real-arm protocol is appropriate, but promised experiments are not
evidence.

**Repair condition:** report every valid/aborted paired nominal versus frozen
global-residual trial on the original five-DoF arm, with selective IPWM as an
optional attribution row, including lock drift, reach, contact,
continuous terminal error, success, camera setup, and failure codes. If method
benefit is absent, use the real arm to validate the six-stage failure diagnosis
rather than claiming recovery.

### M3 — Fresh-query confirmation does not confirm a large effect

**Severity:** major. **Criterion:** technical soundness and evidence.

D3 was historically inspected and therefore cannot provide pristine domain
confirmation. The project nevertheless registered a new candidate seed before
generation and evaluated all three frozen checkpoints once. Global residual
improves success in 3/3 seeds but regret by only 9.77% on average with 2/3
positive, missing the predeclared moderate and strong gates. This bounds rather
than validates transfer of the 19.76% development effect.

**Repair condition:** retain the negative query result and do not tune on D3.
Obtain independent evidence through the frozen physical protocol or a truly
new setting chosen before inspection, with diagnosis rather than dominance as
the primary hypothesis.

### M4 — Limited task breadth and weak success movement

**Severity:** major. **Criterion:** significance and evidence.

The stable 19.76% regret reduction translates to only 4.04% terminal-error
reduction and +1.58 percentage points success, with one seed tied on success.
The evidence is informative, but does not establish broad recovery capability.

**Repair condition:** prioritize continuous paired terminal error and regret;
do not center binary success. Add a second prespecified fault severity, target
family, or fixed-pregrasp feasibility panel only if it uses the frozen method
and does not displace the real push experiment.

## Writing and presentation concerns

### Writing scorecard

| Dimension | Weight | Score | Confidence | Evidence basis | Concrete repair |
|---|---:|---:|---:|---|---|
| Storyline and motivation | 10 | 4 | 4 | Abstract and revised introduction | Keep the six-stage question as the sole main line. |
| Contribution display | 10 | 3 | 4 | Contributions still mix framework, projection, comparison, and result | State one primary diagnostic contribution, one structural mechanism, and one empirical finding. |
| Paragraph logic | 10 | 3 | 4 | Historical results occupy multiple subsections after the new primary result | Move most historical matrices to a compact failure-boundary table. |
| Claim-evidence alignment | 12 | 4 | 5 | Abstract/table/conclusion numbers match JSON; limitations are explicit | Add artifact IDs or appendix pointers for every primary row. |
| Method readability | 9 | 4 | 4 | State-isolation figure plus six-stage evidence chain | The evaluation flow is now visible; keep the global/selective distinction explicit in the caption and method text. |
| Experiment narration | 9 | 4 | 4 | Primary table is interpreted before historical material | Add confidence intervals and a visual showing where the metric tradeoff arises. |
| Related-work positioning | 8 | 3 | 3 | Short technical comparison, limited experimental relation | Replace broad listing with closest-work axes. |
| Terminology consistency | 8 | 3 | 4 | SI-IPWM, selective IPWM, state isolation, global residual, carrier, and historical router coexist | Freeze one name per formal row and label historical models as retrospective at first mention. |
| Prose discipline and voice | 10 | 3 | 4 | Honest but defensive phrases such as “disclosed failure” recur | Retain limitations but state the positive scientific question before development history. |
| LaTeX and format discipline | 6 | 4 | 5 | Compiles, seven of eight pages, tables fit, references resolve | Switch to the official double-anonymous class/options and clear PDF metadata before submission. |
| Reviewer-facing risk | 8 | 4 | 5 | Primary insight is visualized; hardware panel remains absent | Add the real setup/results panel without weakening the six-stage figure. |

**Weighted writing score: 3.54/5. Writing risk: moderate.**

## Format and venue concerns

- Official ICRA 2027 allows eight total pages, including references. The current
  seven-page PDF passes the length check and leaves approximately one page for
  the six-stage figure and real-arm panel.
- The source uses `\documentclass[conference]{IEEEtran}` rather than an explicit
  double-anonymous review option. “Anonymous Authors” hides names visually, but
  class/options and PDF metadata still need a final policy audit.
- The paper contains no external URLs, acknowledgments, or visible affiliations,
  but repository/archive metadata have not been checked for identifying names.
- The new six-stage figure now visualizes the strongest validated contribution;
  a real-arm setup/results panel remains missing.
- The paper compiles with no undefined references or overfull horizontal boxes.

**Desk rejection risk: low to medium.** The current length and readable PDF are
safe, but final anonymity/template compliance remains unverified.

## Multi-reviewer panel

### Best-justified reviewer

- **Likely score:** 6/10, weak accept; **confidence:** 4/5.
- **Positive signal:** a rare honest, controlled demonstration that prediction
  RMSE and action utility diverge, with exact constraint enforcement and a
  reusable staged diagnosis.
- **Negative signal:** scope is narrow and physical validation is absent.
- **Score-change condition:** valid real-arm failure-stage evidence, even without
  a large method win, could sustain 6 because the negative D3 query is retained.

### Critical reviewer

- **Likely score:** 4/10, reject; **confidence:** 5/5.
- **Positive signal:** unusually transparent ablations.
- **Negative signal:** the named selective innovation loses to a generic matched
  residual, success gain is tiny, and the D3 query misses its preregistered gate.
- **Fatal concern:** novelty may collapse to “hard projection plus residual
  training and a diagnostic checklist.”
- **Score-change condition:** mechanism-specific confirmation or multi-setting
  evidence that makes the diagnostic framework itself a substantive result.

### Method and soundness reviewer

- **Likely score:** 5/10; **confidence:** 4/5.
- **Positive signal:** exact projection and paired regret are well defined.
- **Negative signal:** the paper does not establish why the global residual
  improves ranking while response RMSE fails, beyond empirical observation.
- **Score-change condition:** add per-stage causal analysis or stratified,
  predeclared evidence linking contact ambiguity to ranking gain.

### Evidence and experiment reviewer

- **Likely score:** 5/10; **confidence:** 5/5.
- **Positive signal:** 400x128x50 matched evaluation and honest No-Go reporting.
- **Negative signal:** the post-freeze D3 query misses its regret gate, the
  manuscript lacks a primary confidence interval, and there is no real robot or
  closed-loop MPC.
- **Score-change condition:** paired statistics and complete real-arm evidence.

### Novelty and positioning reviewer

- **Likely score:** 4/10; **confidence:** 3/5.
- **Positive signal:** six-stage fault-control diagnosis is a useful framing.
- **Negative signal:** the literature section does not yet prove that this exact
  evaluation decomposition or joint-lock setting is underexplored.
- **Score-change condition:** current close-work audit and technically explicit
  differentiation, without first/SOTA claims.

### Writing and clarity reviewer

- **Likely score:** 5/10; **confidence:** 5/5.
- **Positive signal:** primary numbers and limitations are easy to recover.
- **Negative signal:** old historical material still dilutes the new paper.
- **Score-change condition:** compress retrospective tables and use the recovered
  space for the real-arm panel.

### Ethics and reproducibility reviewer

- **Likely score:** 7/10; **confidence:** 5/5.
- **Positive signal:** provenance ledger, invalid-run disclosure, tests, compute
  ledger, and explicit missing-evidence boundaries.
- **Negative signal:** end-to-end training package and immutable artifacts are
  incomplete.
- **Score-change condition:** archive hashes, environment lock, data/checkpoint
  availability, and real-arm safety ledger.

### AC or meta-review synthesis

- **Agreement:** the protocol and transparency are strengths; selective
  attribution fails, D3 transfer is weak, and physical evidence is missing.
- **Disagreement:** whether the six-stage diagnosis plus regret/RMSE decoupling is
  itself sufficiently novel for ICRA.
- **Decisive accept axis:** demonstrate that the diagnosis transfers beyond one
  development setting and is physically meaningful on the original arm.
- **Decisive reject axis:** the only task-level gain is reproduced by a generic
  global residual and remains simulation-development evidence.
- **Final calibrated stance:** 5/10, weak reject/borderline.

## Concern-to-action table

| Priority | Concern | Required action | Evidence that closes it | Expected movement |
|---:|---|---|---|---|
| P0 | No real-arm evidence | Execute frozen low-speed paired push protocol; retain aborts and both videos | Validity ledger, raw hashes, paired terminal/contact/lock table, setup panel | Evidence +0.5 to +1 dimension; overall may move 5→6 if coherent |
| Done | No post-freeze query confirmation | Registered and ran D3 seed 91031 once with frozen checkpoints; retained the failed gate | Candidate SHA-256, per-seed rows, unchanged model checkpoints | Raises confidence/auditability but not performance score |
| P0 | Selective attribution failed | Reframe primary contribution as diagnostic unless new frozen evidence changes it | Abstract/introduction/method/figure all agree; no hidden dominance language | Prevents score loss rather than creating novelty |
| Done | Six-stage insight was not visualized | Added a compact six-stage pipeline with per-stage metrics and pass/No-Go markers | Legible vector Fig. 2 in the compiled PDF | Clarity increased from 3 to 4 |
| P1 | Historical evidence dominates space | Compress GenkiArm/simplified matrices into one retrospective boundary table | At least half a page recovered for primary evidence | Writing risk moderate→low |
| P1 | Primary uncertainty absent in PDF | Add seed rows and paired/bootstrap interval or explicitly descriptive range | Machine-generated table/plot matching JSON | Evidence confidence increases |
| P2 | Packaging incomplete | Archive formal checkpoints and environment lock; run clean-machine end-to-end training/evaluation | Immutable checkpoint manifest and successful clean run | Reproducibility 4→5 |
| P2 | Anonymity mode unverified | Use official double-anonymous template/options and inspect metadata | Submission PDF desk-check log | Desk risk low |

## Score-change conditions

| Change | Condition | Likely affected dimensions | Expected movement |
|---|---|---|---|
| Raise score | Valid real-arm paired evidence, retained negative D3 query, six-stage figure, and coherent diagnostic framing | Evidence, soundness, clarity, significance | Overall 5→6 is plausible; project score approaches 3.9--4.1/5 |
| Lower score | Real-arm trials contradict even the constraint/diagnostic story, confirmation reverses regret direction, or a close paper already provides the same six-stage contribution | Evidence, novelty, soundness | Overall 5→4 or lower |
| No quick change | Making selective IPWM a strong novel mechanism without new evidence | Novelty | Rhetorical editing alone cannot raise the score |

## Recommended next owner

Experiment execution first: real-arm validity packet. The D3 query confirmation
and six-stage figure are complete. Then compress historical material and perform
final submission/anonymity checking.

## Checks run

- Read the complete seven-page PDF and the active LaTeX source.
- Verified the four primary machine-readable summaries and provenance ledger.
- Regenerated all primary and D3-query summaries with the one-command script;
  60 focused tests passed.
- Independently regenerated the 25,600-row D3 archive and obtained the exact
  registered SHA-256 `43a00365...72e2702`.
- Verified GitHub `main` and the working branch at commit `8d49a67`.
- Compiled the PDF and visually inspected all pages; no clipped table or
  unresolved reference remains.
- Checked the official ICRA 2027 call for papers for deadline, page limit,
  anonymity, and video constraints.
- Searched the source/checklist for stale “untouched confirmation” language and
  corrected the submission checklist.

## Unresolved or unverified

- No real-arm trial data or setup photograph.
- No pristine unseen-domain confirmation; the available post-freeze D3 query
  is explicitly non-pristine and misses its regret gate.
- No independent clean-machine end-to-end training reproduction.
- No final PDF metadata/anonymity audit against PaperPlaza's exact template.
- Novelty relative to the most recent 2025--2026 fault-aware world-model papers
  has not been re-searched in this review; positioning confidence is therefore
  3/5 rather than 5/5.

## Output self-check

- Scores, confidence, and decision stance are separated.
- Every score of 3/5 has a deduction and repair condition.
- No acceptance probability is claimed.
- Positive, negative, and missing evidence remain distinct.
- The 19.76%, 4.04%, 270.04%, projection-drift, reachability, and contact values
  in Fig. 2 match the frozen JSON.
