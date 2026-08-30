# GenkiArm confirmation V2: seed-107 core matrix

## Status

Seed 107 is the first completed preregistered run on the calibrated
`sim/assets/genkiarm_push.xml`. It uses disjoint query seed 1107, 10 paired
trajectories per condition, horizons 10/25/50, and the three preregistered
primary domains. The final artifact contains 84 aggregate cells and 6,720 raw
paired records.

This is a single training seed. It is a diagnostic mechanism result, not a
population-level significance claim and not the five-seed paper result.

## Paired primary-domain effects versus mechanism-matched carrier

| Domain | Method | Object RMSE improvement | Free-joint regression | Locked RMS |
|---|---|---:|---:|---:|
| high damping | full-state IPWM | +1.2348% | +0.2007% | 0 |
| high damping | selective IPWM | +1.2348% | 0.0000% | 0 |
| high damping | routed selective | 0.0000% | 0.0000% | 0 |
| mixed composition | full-state IPWM | +1.3443% | -0.2821% | 0 |
| mixed composition | selective IPWM | +1.3443% | 0.0000% | 0 |
| mixed composition | routed selective | 0.0000% | 0.0000% | 0 |
| mixed unseen | full-state IPWM | +1.6475% | +0.4164% | 0 |
| mixed unseen | selective IPWM | +1.6475% | 0.0000% | 0 |
| mixed unseen | routed selective | +1.6475% | 0.0000% | 0 |

Across the three primary domains and three horizons:

- raw selective IPWM: **+1.4198% object improvement, 0% free-joint
  regression**;
- routed selective IPWM: **+0.6037% object improvement, 0% free-joint
  regression**;
- preregistered object-effect threshold: **at least +5%**.

## Interpretation

The selective rollout isolates the free robot state exactly while preserving
the object benefit of the full-state intervention. This is direct evidence for
the state-isolation mechanism relative to full-state IPWM: on high damping and
mixed unseen, it removes +0.2007% and +0.4164% free-state regressions without
giving up object improvement.

The result does **not** establish a practically or statistically significant
gain over the carrier. The effect is positive but small and below the frozen
5% gate. The guard is conservative: it falls back on high damping and mixed
composition, and enables the intervention only on mixed unseen among the three
primary domains. Consequently, routing reduces the already-small raw effect.

The physical-context stage selected epoch 0, so seed 107 provides no evidence
that the newly trained physical-context parameters improve generalization. The
observed object benefit comes from the frozen contact-residual template; the
new contribution supported here is selective state isolation, not physical-
context adaptation.

## Decision

Seed 107 is **No-Go for the full preregistered performance gate** and **Go for
the narrow state-isolation invariant/mechanism check**. Four additional fresh
training seeds are required before any five-seed statement. A one-seed
trajectory bootstrap is not a substitute for training-seed uncertainty.

Authoritative artifacts:

- `runs/g2_ipwm_genkiarm_confirmation_v2/seed107_v1/metrics.json`
- `runs/g2_ipwm_genkiarm_confirmation_v2/seed107_v1/raw.json`
- `runs/g2_ipwm_genkiarm_confirmation_v2/seed107_v1/seed107_diagnostic_summary.json`
