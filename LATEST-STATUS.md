# Latest project status

Last updated: 2026-08-31

## Frozen question and platform

The paper studies few-shot world-model adaptation after a diagnosed joint lock,
and when prediction changes do or do not transfer to contact-action selection.
The primary platform is the original five-DoF arm model in
`sim/assets/arm_push.xml`; GenkiArm and Panda results are transfer or feasibility
boundaries and cannot replace primary-platform evidence. The XML is a precise
kinematic task model, not a fully identified dynamic digital twin.

The frozen method family is analytic lock projection, a fault residual, and
paired counterfactual decision training. The strict protocol uses D2/D4
development, seeds 7/17/27, 400 groups per seed, 128 candidates per group, and
50-step rollouts. D3 was excluded from fitting but historically inspected, so it
is not described as untouched confirmation.

## Current primary result

| Comparison | Response RMSE | Spearman | Top-1 regret | Terminal error | Success |
|---|---:|---:|---:|---:|---:|
| Same-capacity global residual vs nominal | -270.04% | +0.0375 | **+19.76% (3/3)** | **+4.04% (3/3)** | +1.58 pp |
| Selective IPWM vs nominal | -247.90% | +0.0378 | +18.28% (3/3) | +3.73% (3/3) | +1.17 pp |
| Selective IPWM vs global residual | -19.52% | +0.00037 | -2.07% | -0.32% | -0.42 pp |

Positive regret/error values mean reductions. The last outcome is the realized
result of the selected open-loop candidate, not receding-horizon MPC.

The stable positive result is **control-related fault adaptation versus nominal**,
not selective IPWM versus a matched alternative. The global residual reproduces
or exceeds the selective result on aggregate. Full-state and selective
publication are exactly identical under the current formal protocol. Selective
or path-support attribution is No-Go.

The realized-cost oracle gives 0.03758 m mean endpoint error versus 0.04675 m
for nominal, i.e. 9.17 mm or 19.63% endpoint headroom with 3/3 positive seeds.
This is a privileged upper bound, not a deployable method or performance claim.

The response result is intentionally negative: better candidate choice appears
while contact-response RMSE becomes much worse. This supports the six-stage
diagnosis that constraint, reachability, contact, response prediction, action
ranking, and realized outcome are distinct gates.

## Post-freeze D3 candidate-query confirmation

Before generating a new archive, the project registered candidate seed 91031,
200 groups, 128 distinct candidates, 50 steps, all three frozen model seeds, and
fixed interpretation thresholds. The archive audit passed with SHA-256
`43a00365caf59e504ef7b730fc9d91bc7bfd0d9efce79899a7b9d725072e2702`.
Because D3 had appeared in historical exploration, this is an untouched-query
confirmation after freezing, not a pristine unseen-domain claim.

On D3, the same-capacity global residual versus nominal gives 9.77% mean regret
reduction (2/3 positive), 2.00% terminal-error reduction (2/3), and +2.17
success percentage points (3/3). Response RMSE worsens 263.63% and Spearman
changes by -0.0148. It therefore misses the preregistered moderate gate of at
least 10% regret reduction with 3/3 direction. Selective IPWM also fails
attribution versus global: regret -3.82%, terminal error -0.50%, and success
-0.83 points on aggregate, each with only 1/3 favorable seeds. No large
transferable task-performance advantage was found.

## Analytic projection result

Removing projection gives maximum lock-position drift of 0.077--0.153 rad
(mean 0.116 rad or 6.63 degrees) and maximum lock-speed violation of
0.384--0.660 rad/s (mean 0.539 rad/s). With projection, both are exactly zero
in 3/3 seeds. This is a structural correctness contribution, not a claim of
large success-rate improvement.

## Decision-loss and historical boundaries

- Weight 10 versus weight zero improves regret by 5.98% and terminal error by
  1.40% on average, but only in 2/3 seeds; success decreases 0.50 pp and response
  RMSE worsens 356.44%. The decision loss is a tradeoff, not a uniformly
  improving component.
- Weight zero versus nominal improves response RMSE by 24.84% and success by
  1.67 pp in 3/3 seeds, but ranking and endpoint directions are inconsistent.
- The constrained-IK carrier remains a strong feasibility baseline; hard mask
  and one-step SFET fail to establish contact reliably.
- Calibrated GenkiArm selective prediction, Panda contact transfer, guarded MPC,
  and method-level grasp remain No-Go or feasibility-only evidence.

## Evidence boundaries

Supported:

- exact lock constraint satisfaction from analytic projection;
- 19.76% mean top-1-regret reduction versus nominal with 3/3 seed direction;
- 4.04% mean selected-candidate terminal-error reduction with 3/3 direction;
- a reproducible six-stage diagnostic and prediction--decision tradeoff.

Not supported:

- selective publication as the cause of the current control-related gain;
- improved response RMSE for the decision-trained model;
- receding-horizon MPC benefit;
- untouched-domain confirmation, cross-arm object/contact transfer, learned
  grasp, visual closed loop, or real-robot method benefit;
- an objective 4+/5 ICRA assessment before real-robot evidence and independent
  review.

The 2026-08-31 independent CCFA/ICRA-style review scores the current seven-page
version at approximately **3.6/5** and **5/10 (weak reject/borderline)**. The new
six-stage evidence figure improves clarity, but real-arm evidence, untouched
confirmation, and novelty positioning remain decision-level gaps.

## Reproduction and next action

The machine-readable sources are:

- `results/final/primary-strict-development-3seed-summary.json`
- `results/final/primary-decision-loss-ablation-3seed.json`
- `results/final/primary-global-matched-ablation-3seed.json`
- `results/final/primary-projection-ablation-3seed.json`
- `results/final/confirmation-d3-query-seed91031-summary.json`
- `results/final/confirmation-d3-query-seed91031-audit.json`
- `reports/primary-result-provenance-ledger-20260831.md`

With formal local run outputs present, regenerate all summaries and focused
tests with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_primary_evidence.ps1
```

The original five-DoF real experiment remains the highest-priority missing
evidence. It uses a gripper to push a block, two fixed eye-to-hand cameras, low
speed, intact/D2/D3 conditions, synchronized raw trajectories, lock/safety
truth, calibration trials, and blind evaluation actions. No real-robot number
may enter the paper until raw logs pass the validity ledger.

The primary paired hardware comparison is now `nominal` versus
`global_matched`, because that is the only matched comparison with a stable
simulation control signal. `si_ipwm` is an optional third row for structural
attribution. The analyzer rejects duplicate rows and mismatched reset positions
and reports paired endpoint, success, reach, and contact differences.

The optional grasp packet is now executable but deliberately narrow: one fixed
pregrasp, short vertical lift, three-second retention, and at most five trials
per intact/D2/D3 condition. Its analyzer preserves aborts and may support only
feasibility, never a learned-grasp or method-dominance claim.

See `PROJECT-PLAN-V6.md` and `paper/main.pdf` for the full plan and manuscript.
