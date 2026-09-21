# IPWM manuscript workspace

The canonical English manuscript is `main.tex` / `main.pdf` (eight pages).
The Chinese companion is `main_zh.tex` / `main_zh.pdf` (seven pages).
Earlier drafts and review reports are historical records.

## Current evidence

The frozen deployed model is selective IPWM, checkpoint
`../runs/icra_confirmation_d3_query_selective_w10/seed27/model.pt`.
Its robot predictor has no contact-conditioned feedback or intervention-context
input. Selective and full-state feedback therefore produce identical
predictions in this final configuration; structural isolation is not an
independently demonstrated performance advantage.

The primary simulation comparison uses three seeds, 400 groups of 128
candidates, and a 50-step horizon. Outcomes evaluate the selected open-loop
candidate. The matched global residual reduces regret by 19.76% against
nominal; selective IPWM does not outperform the matched global model.
Prediction accuracy and ranking quality must be discussed separately.
The D3 query confirmation is weaker; D3 was historically inspected.

Hardware evidence comprises 18 core trials (intact and D1--D5, three each)
and five double-lock trials. Core success is 18/18. Double-lock success is
1/1, 1/1, and 1/3 for J1+J2, J2+J3, and J3+J4 respectively. The controller
replans using measured joints and visual object feedback, ranks 10,000
candidates over 50 steps, and commands reference index 24 each cycle.
Candidate count is planning computation, not physical trial count.
No matched hardware baseline establishes comparative superiority.

## Audit and revision records

- `numeric-consistency-check.json`: primary numerical comparisons and projection ablation.
- `historical-numeric-check.json`: historical prediction tables recomputed from per-window errors.
- `genkiarm-bootstrap-check.json` and `genkiarm-bootstrap-recomputed.json`: calibrated-model bootstrap reproduction.
- `checkpoint-correspondence.json`: checkpoint tensor and manifest correspondence.
- `hardware-controller-check.json` and `hardware-summary-check.json`: controller and 23-trial audit.
- `training-data-check.md`: configured training counts and historical lineage limits.
- `training-cache-discovery.json` and `training-stage-correspondence.json`: cache inventories and differences.
- `neighbor-paragraph-comparison.md`: comparison with ActivePusher, without claiming a public highest review score.
- `citation-verification.md`: primary-source reference checks.
- `simulation-comparison-gap.md`: missing simulation comparison using the final deployed controller.
- `ccfa-review-reports/current-icra-review.md`: substantive review and remaining weaknesses.
- `venue-policy.md`: submission constraints, including the eight-page initial limit.
- `polishing-plan.md`: revision history and remaining work.

Historical cache versions contain different trajectories. A unique sample
total across every predecessor training stage has not been established.
Additional experimental evidence remains separate from manuscript edits.

## Build

From this directory: run `pdflatex main.tex`, `bibtex main`, then two more
`pdflatex main.tex` passes. For Chinese use `xelatex main_zh.tex` and
`bibtex main_zh`. Check references, page count, and rendered layout.
No submission has been performed.
