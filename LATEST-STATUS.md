# Latest project status

Last updated: 2026-08-30

## Current empirical decision

The old large positive Push result was obtained on the simplified
`sim/assets/arm_push.xml` development model.  It is not evidence on the
calibrated-kinematic GenkiArm model and must not be presented as such.

Fresh preregistered GenkiArm seeds 107/117/127 are complete.  Routed selective
SI-IPWM improves object RMSE by +0.6037%, -0.7836%, and +0.7182%, respectively
(mean +0.1794%; 2/3 positive; 95% interval crosses zero).  Raw selective
SI-IPWM averages +0.6841%.  Both retain zero free-state regression and zero
lock violation.  The current decision is therefore:

> **Performance superiority No-Go; analytic constraint and selective
> state-isolation mechanism replicated.**

Seeds 137 and 147 remain required by the frozen confirmation protocol.  No
4+/5 ICRA claim is authorized by current evidence.

## Evidence boundaries

- Hard lock projection and the state-isolation invariant are supported.
- Stable object-prediction superiority is not supported.
- Closed-loop control superiority is not supported.
- Panda object/contact transfer is No-Go (0/3); Panda scripted grasp 5/5 is
  task feasibility only.
- Dual fixed eye-to-hand visibility passes the frozen perturbation audit but is
  observability evidence, not visual world-model evidence.
- There are no real-robot results in the paper yet.

## Current research direction

The 2025--2026 literature audit ranks DyWA (ICCV 2025), SimDist (RSS 2026), and
PIN-WM (RSS 2025) as the closest successful systems.  Their mechanisms are to
be reproduced only as strong baselines/root-cause diagnostics.  They must not
be added to SI-IPWM and relabelled as original contributions.  SI-IPWM survives
as a paper core only if projection, selective isolation, and guard provide an
independent, capacity-matched increment over a reproduced strong adaptation
baseline.

See:

- `reports/genkiarm-three-seed-interim-20260830.md`
- `reports/closest-top-conference-methods-2025-2026-20260829.md`
- `reports/genkiarm-evidence-ledger-and-v2-freeze-20260829.md`
- `PROJECT-PLAN-V6.md`

