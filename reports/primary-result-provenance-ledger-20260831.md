# Primary result provenance ledger — 2026-08-31

This ledger is the mandatory claim boundary for the ICRA paper. A result may be
used only for the model and protocol named in its row. Similar names, shared
losses, or shared projection code do not make two models equivalent.

| Evidence | Exact model identity | Data / evaluation | Verified result | Permitted claim | Prohibited attribution | Status |
|---|---|---|---|---|---|---|
| `metric-push-goal-summary.json` in the secondary workspace | Compact sequence model that predicts all 14 state dimensions over five segments, with analytic lock projection and sequence soft-regret training; **no shared carrier and no selective publication support set** | Frozen 128-candidate D3 evaluation, 300 paired episodes, rollout seeds 307/317/327 | endpoint error -8.93%; success +10.33 pp; top-1 regret -26.97%; zero lock violation; 3/3 directions agree | Analytic projection plus decision-focused sequence training can improve ranking and closed-loop outcomes under this protocol | The gain was caused by selective IPWM, path-supported publication, or the authoritative SI-IPWM checkpoint | Positive, model-specific; D3 was held out from final fitting but is not pristine never-inspected data |
| `runs/icra_primary_decision_full_w10/seed7/summary.json` | Authoritative carrier + private intervention branch + selective publication + analytic projection + paired decision loss, weight 10 | D2/D4 development; 320 train groups, 80 validation groups, 32 candidates; 40 epochs | selected epoch 32; all 320 training groups seen; validation Spearman increased from about 0.0164 to 0.0636 and regret decreased about 4.2% | The paired loss can fit the development candidate distribution | Generalization, D3 confirmation, or 128-candidate control improvement | Development-only signal |
| `runs/icra_primary_decision_full_w10_128eval/seed7/summary.json` and the first six-stage reruns | **Invalid evaluation wiring:** `--initialize-candidate-model` loaded only 317,834 robot-prefixed parameters from the 337,834-parameter checkpoint; object and intervention heads remained newly initialized with zero evaluation epochs | D2/D4; 400 x 128 x 50 | The reported carrier/full/selective ties and response divergence describe the incorrectly reconstructed model, not epoch 32 | Evidence of a checkpoint-loading bug and reason for strict full-state loading | Any claim about the selected checkpoint's independent performance | **INVALID — superseded by strict full-checkpoint rerun** |
| weight-1/10/50 short pilots under `runs/icra_primary_decision_pilot_*` | Same authoritative architecture, short decision-loss development pilots | D2/D4, 32-candidate development protocol | weight 1 selected epoch 0; weight 10 and 50 improved validation decision metrics but did not establish independent 128-candidate benefit | Hyperparameter pilots and stopping evidence | Paper-table performance or confirmation | No-Go / exploratory only |
| `runs/ipwm_128candidate_full_eval_diagnostic_20260831/seed7/summary.json` | Epoch-0 diagnostic with robot-only initialization semantics | D2/D4; 400 x 128 x 50 | carrier/full/selective tie | Historical pipeline diagnostic only | Selected-checkpoint or formal performance | Superseded |
| Historical simplified-arm selective-prediction audits | Exact historical SI-IPWM variants on simplified `arm_push.xml` | Multiple seeds/domains/horizons, prediction metrics | object-RMSE signal and exact carrier-relative state isolation in the audited cells | Structural isolation and simplified-domain prediction evidence | Original 5-DoF, calibrated GenkiArm, real-robot, or closed-loop dominance | Supporting evidence only |
| Calibrated GenkiArm seeds 107/117/127 | Raw/routed selective SI-IPWM on fresh provenance-corrected GenkiArm data | Three seeds, prediction evaluation | mean object improvements are small and inconsistent (2/3 positive); intervals cross zero | Cross-arm feasibility and honest negative boundary | Stable cross-arm advantage or confirmation success | No-Go for performance |
| `results/final/primary-strict-development-3seed-summary.json` | Strictly loaded 337,834-parameter carrier/full/selective architecture with analytic projection and weight-10 paired decision training | D2/D4 development seeds 7/17/27; each 400 groups x 128 candidates x 50 steps; V2 continuous contact labels | Spearman delta positive 3/3, mean +0.02023; regret reduction mean 4.44% (2/3); endpoint reduction mean 0.796% (2/3); success mean -0.25 pp; response RMSE mean -18.07%; zero violations; full-state=selective 3/3 | Small directional action-ranking signal plus exact structural feasibility; magnitude and path-support attribution fail | Stable task-performance advantage, 10% endpoint gain, or selective-publication contribution | **Directional signal; magnitude/attribution No-Go** |
| `results/final/primary-decision-loss-ablation-3seed.json` | Same architecture and 40-epoch protocol; paired comparison of decision weight 10 versus 0 | D2/D4 seeds 7/17/27; independent V2 400 x 128 x 50 | weight10 vs weight0: regret +5.98% mean (2/3), endpoint +1.40% (2/3), but success -0.50 pp, Spearman delta -0.0108 and response RMSE -356.44%; weight0 vs nominal improves response 24.84% and success +1.67 pp, both 3/3 | Soft-regret shifts some performance toward selected-action regret/endpoint but is not a stable all-stage improvement | Decision loss explains the full nominal-baseline gain or robustly improves success | **Directional regret effect; response/success No-Go** |
| `results/final/primary-global-matched-ablation-3seed.json` | Same-capacity global residual with analytic projection and weight-10 decision training versus nominal and selective IPWM | D2/D4 seeds 7/17/27; V2 400 x 128 x 50 | versus nominal: regret -19.76% mean (3/3), endpoint -4.04% (3/3), success +1.58 pp; response RMSE worsens 270.04%. IPWM minus global: regret -2.07%, endpoint -0.32%, Spearman +0.00037 | Control-related adaptation can improve candidate choice despite severe response-RMSE degradation | Existing ranking gain is uniquely caused by selective/path-supported IPWM | **Stable regret result; selective structural attribution No-Go** |
| `results/final/primary-projection-ablation-3seed.json` | Frozen full checkpoint evaluated with versus without analytic lock projection | Same strict three-seed V2 protocol | no projection: maximum lock-position drift 0.077--0.153 rad (mean 6.63 degrees) and velocity violation 0.384--0.660 rad/s; projection: exact zero for both in 3/3 seeds | Analytic projection supplies exact structural constraint satisfaction that the learned rollout does not | Projection itself produces a large task-success gain | **Constraint-satisfaction Go; task-gain claim excluded** |
| `results/final/confirmation-d3-query-seed91031-summary.json` | Frozen seed 7/17/27 selective IPWM and same-capacity global residual checkpoints; no retraining | Post-registration D3 candidate seed 91031; 200 groups x 128 candidates x 50 steps; archive hash fixed before evaluation | global vs nominal: regret -9.77% (2/3), terminal error -2.00% (2/3), success +2.17 pp (3/3), response RMSE worsens 263.63%; selective vs global fails regret/endpoint/success attribution | A fresh candidate-query sample gives a small consistent success-rate direction and bounds transfer of the development effect | Pristine unseen-domain confirmation, a >=10% 3/3 regret advantage, or selective-IPWM dominance | **Fresh-query confirmation; large-advantage and attribution No-Go** |

## Dataset contamination and confirmation rule

- Earlier development configurations accidentally named D3 as `primary_domain` and
  computed D3 post-training prediction diagnostics. D3 was not used for gradient
  fitting or epoch selection in those runs, but it has been repeatedly inspected.
- Therefore D3 may be described only as **held out from final fitting with fresh
  rollout seeds**, not as a pristine never-seen confirmation domain.
- No D3 metric emitted by the weight sweep may enter a confirmation table.
- A new confirmation set cannot be substituted silently because the current goal
  freezes D3. Any replacement requires an explicit protocol amendment before data
  generation and must be disclosed as such.
- The seed-91031 D3 archive was registered before generation and evaluated once
  after checkpoint freeze. Its untouched unit is the candidate/query sample, not
  the D3 domain itself.

## Current paper-safe conclusion

The strongest coherent conclusion is diagnostic rather than a selective-IPWM
victory claim: analytic projection eliminates 3/3 learned-rollout lock violations,
while control-related adaptation lowers top-1 regret 19.76% against nominal in
3/3 seeds even though response RMSE becomes much worse. The same-capacity global
residual reproduces or exceeds the selective model on aggregate, so the control
gain cannot be attributed to selective publication. Hard feasibility, predictive
accuracy, contact establishment, response prediction, action ranking and realized
candidate outcomes must remain distinct gates.
