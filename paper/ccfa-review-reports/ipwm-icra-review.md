# IPWM ICRA 严格审稿基线

- Review date: 2026-08-28
- Manuscript: `paper/main.pdf` / `paper/main.tex`
- Review mode: full scientific + writing + format
- Venue assumption: IEEE ICRA contributed paper, 6 pages technical content plus references
- Evidence read: six-page English PDF, LaTeX source, G2-R audit report and summary, PROJECT-PLAN-V6, G3 protocol
- Current weighted score: **3.1/5**
- Calibrated stance: **weak reject**
- Confidence: **4/5**

## Paper summary

The paper proposes IPWM for few-shot prediction adaptation after a diagnosed joint lock in planar pushing. Analytic projection enforces the locked coordinate, while a frozen contact-aware robot block, explicit pusher geometry, a low-rank object intervention residual, observable physical context, and rollout-depth scheduling address unknown downstream interaction changes. The strongest evidence is a matched five-seed simulation audit with 30 held-out object cells, paired trajectory-cluster intervals, structural ablations, zero lock violation, and an immutable artifact trail. The current PDF is not submission-ready because it contains explicit real-robot TODOs, an empty results table, a placeholder setup figure, an incorrect eye-in-hand hardware description, and an underfilled reference page.

## Quantitative scorecard

| Dimension | Score (1-5) | Confidence | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 3 | Sec. II-IV; Table I | The diagnosed-lock-to-object-intervention interface is plausible and distinct, but closest positioning is thin and several components are established ideas. Reach 4 by sharpening the single novelty delta and comparing against the closest few-shot pushing/adaptive-dynamics methods under a claim-matched protocol. |
| Soundness | 4 | 4 | Sec. IV; Eq. 1-3; Sec. V-A/F | Analytic feasibility and frozen matched protocol are credible. The 5.97% free-joint miss and incomplete causal isolation prevent a clear-strength score. Reach 5 only with a pre-frozen diagnosis or mitigation that survives new seeds. |
| Evidence | 3 | 5 | Tables IV-VII; Fig. 2; Sec. V | Five seeds, paired intervals, and ablations are good, but all task evidence is simulation prediction rather than closed-loop pushing. The PDF visibly promises missing real-arm evidence. Reach 4 without hardware by replacing the unrealized real-arm claim with substantially broader simulation: closed-loop task success, multiple object/contact regimes, stronger baselines, failure analysis, and untouched confirmation seeds. |
| Significance | 3 | 4 | Abstract; Introduction; limitations | Joint-lock adaptation is relevant but narrow, and the current evidence does not yet show planning/control benefit. Reach 4 by demonstrating that prediction gains translate into robust closed-loop task outcomes across held-out locks and physics. |
| Clarity | 2 | 5 | Pages 1, 4, 5, 6 | Red TODOs, placeholder figure/table, incorrect eye-in-hand description, dense method packaging, and a mostly blank reference page make the manuscript visibly unfinished. Reach 4 by removing all prospective hardware material, rebuilding the six-page story around a simulation-only claim, and adding inspectable failure/closed-loop figures. |
| Reproducibility | 4 | 4 | Sec. V-D; audit summary; hashes/tests | Raw rows, seeds, configs, hashes, tests, and frozen protocols are unusually strong. Reach 5 by exposing a single clean reproduction entry and generating all paper tables/figures from audited artifacts. |
| Ethics / limitations | 4 | 5 | Sec. V-E/F; Sec. VI | The paper discloses the free-joint miss and avoids interpreting raw current as force. It must remove prospective safety language that can be mistaken for validated hardware evidence. |

Weighted ICRA score: **3.1/5**. The strongest unresolved concern is evidence-to-claim mismatch: an ICRA manipulation paper claims deployment relevance but currently demonstrates only predictive simulation and leaves explicit hardware placeholders.

## Top strengths

1. Exact analytic lock feasibility is simple, auditable, and consistently verified at zero violation.
2. The matched baseline receives projection, K25 support, rank-8 adaptation, identical rollout budgets, and paired trajectories.
3. The expanded five-seed audit includes raw-window data, paired cluster bootstrap intervals, immutable hashes, and negative regimes rather than only favorable aggregates.
4. The paper openly reports the 5.97% free-joint gate miss and avoids a universal-dominance claim.

## Major concerns

| Severity | Concern | Evidence | Repair condition | Owner |
|:---:|:---|:---|:---|:---|
| Fatal for current PDF | Submission contains explicit unfinished placeholders | Abstract; Fig. 3; Table VIII; Sec. V-G; Conclusion | Remove all real-robot TODOs and rebuild the paper as a complete simulation study | paper writing |
| Major | No closed-loop task evidence | Sec. VI admits evidence is predictive | Add frozen-controller simulation experiments reporting success, final error, abort/infeasible rate, and compute | experiment design + execution |
| Major | Novelty delta is not defended against strongest adaptive pushing work | Table I and Related Work use broad binary columns | Add mechanism-level comparison and at least one stronger reproduced baseline where feasible | literature + experiments |
| Major | One free-joint gate fails | Table V; Sec. V-B/E | Add per-step/per-joint failure analysis; either bound task impact or introduce a pre-frozen safeguard validated on new seeds | analysis + experiments |
| Moderate | Component story is dense and partly additive | Sec. IV-B-D; Table VII | Reframe around one invariant, one intervention residual, one reversible risk rule; connect every component to a measured failure mode | paper writing |
| Moderate | Evaluation breadth is narrow | Single simulator/task family, one held-out lock | Add held-out contact/object regimes and at least one additional held-out lock protocol if mechanically valid | experiment execution |

## Desk checks

| Check | Status | Evidence | Consequence / action |
|:---|:---:|:---|:---|
| Length | Pass with poor use of space | Six pages total; page 6 largely empty | Reallocate the page budget to experiments, failure analysis, and references |
| Topic compatibility | Pass | Robot manipulation, failure adaptation, learned dynamics | ICRA fit is credible |
| Minimum quality | Fail | Red TODOs, empty table, placeholder figure | Current PDF is not review-ready |
| Anonymity | Pass/uncertain | Anonymous author line; local artifact URLs must be checked | Run final metadata and URL audit |
| Ethics/reviewability | Pass with limitation | No human subjects; safety claims are mostly bounded | Remove unvalidated real-hardware safety implications |
| Hidden manipulation | Pass | No reviewer/LLM instructions detected in rendered text | No action |

Desk rejection risk: **high for the current PDF**, because the manuscript visibly identifies itself as incomplete. This is fully fixable before submission.

## Multi-reviewer panel

| Reviewer | Tendency | Main positive signal | Main negative signal | Score-change condition |
|:---|:---:|:---|:---|:---|
| Best-justified | borderline | Exact constraint plus unusually careful matched audit | Narrow simulation-only claim | Complete simulation-only story with task outcomes |
| Critical | reject | Transparent limitations | Empty hardware evidence and incomplete PDF | Remove placeholders and close the evaluation loop |
| Method/soundness | borderline positive | Projection and protocol are credible | Free-joint miss and complex residual stack | New-seed safeguard or bounded task-impact evidence |
| Evidence/experiment | reject | Five-seed paired statistics | No closed-loop planning/control result | Add pre-frozen closed-loop simulation matrix |
| Novelty/positioning | weak reject | Damage-intervention interface may be new | Closest adaptive pushing comparison is shallow | Public prior-art audit plus mechanism-matched baseline |
| Writing/clarity | reject | Main method diagram is readable | Manuscript is visibly unfinished | Full six-page rewrite and visual QA |
| Reproducibility | weak accept | Raw rows, hashes, configs, tests | Reproduction entry is fragmented | One-command paper artifact regeneration |

Panel agreement: the simulation audit is credible, but the submitted artifact is incomplete and the evidence does not yet establish task-level robotics value.  
Decisive positive axis: exact feasibility plus matched, auditable few-shot evaluation.  
Decisive negative axis: incomplete paper and prediction-only evidence.  
AC stance: weak reject at 3.1/5; a simulation-only 4+/5 paper is possible if it becomes a complete, task-level, claim-matched study rather than a draft awaiting hardware.

## Score-change conditions

| Change | Condition | Likely dimensions | Expected movement |
|:---|:---|:---|:---|
| Raise | Remove all real-robot placeholders and rewrite as a complete simulation-only paper | Clarity, ethics, evidence | +0.3 to +0.5 |
| Raise | Add frozen closed-loop pushing results with matched baselines, task success, final error, failures, and compute | Evidence, significance, soundness | +0.5 to +0.8 |
| Raise | Add a defensible closest-baseline comparison and sharpen novelty delta | Novelty, significance | +0.2 to +0.4 |
| Raise | Explain or safely bound the free-joint miss on untouched seeds | Soundness, evidence | +0.2 to +0.4 |
| Lower | New closed-loop results show object prediction gain does not improve task outcomes | Evidence, significance | -0.5 to -1.0 |
| Fatal | Any table cannot be reproduced from frozen artifacts or uses post-hoc seed/threshold selection | Soundness, reproducibility | reject |

## Goal acceptance gate

The active improvement goal is complete only when all conditions hold:

1. Independent CCFA/ICRA re-review scores the final English PDF at **at least 4.0/5**.
2. No real-robot result, photo, success claim, or safety certification is used or implied.
3. No TODO, placeholder, empty result cell, incorrect camera architecture, or prospective experiment prose remains.
4. Central claims are backed by frozen simulation artifacts, raw rows, confidence intervals, and reproducible scripts.
5. Closed-loop task evidence and the free-joint failure boundary are visible in the main paper.
6. The six-page PDF passes visual, anonymity, citation, and source-integrity checks.

## Checks run

- Read all six rendered PDF pages and extracted text.
- Cross-checked the principal values against the G2-R audit report and plan.
- Applied the CCFA universal rubric, reviewer panel, calibration, desk checks, and ICRA venue guide.
- Ran a public-safe nearest-work search for adaptive/few-shot pushing, residual physics, and ICRA page policy.

## Unresolved

- Current-year initial-submission ICRA policy must be rechecked immediately before submission.
- The closed-loop simulation capability and available planner/controller baselines still need code-level inspection.
- The free-joint 5.97% miss needs raw per-joint/per-depth diagnosis before deciding whether a model change is warranted.
