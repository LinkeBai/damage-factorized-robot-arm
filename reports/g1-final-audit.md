# G1 Final Audit

## Status

**G1 original gate: complete_no_go. G1-Pivot hybrid: functional pass.**

## Completed

- Reach, D2/D3, state observation, passive calibration and K=0/1/2/5 protocol.
- Three-seed four-method prediction benchmark: topology-only, residual-only,
  monolithic matched and factorized DFWM.
- MuJoCo environment, smoke test, dataset generator, conditional world model,
  latent optimization, frozen actor/MPC and prediction error outputs.
- Reproducibility manifests generated for completed G1 benchmark and control runs.
- Hybrid IK/PD and world-model-assisted hybrid runs are complete:
  world-model hybrid 48/48 successes; option selector 24/24 successes.

## Not passed

- Original learned-MPC frozen-control gate: unstable and mostly unsuccessful.
- Prediction improvement did not translate into a reliable direct learned-MPC
  controller.
- Four methods do not yet have a fully matched frozen-control matrix; therefore
  the original control Go claim is not made.

## Decision

G1 is closed as **complete_no_go with a validated Hybrid Pivot**. G2 must not be
claimed as started under the original direct-control hypothesis. Further work
belongs to V6: option selection, uncertainty-aware rejection, multi-step model
training, and strict ablations against the IK/PD baseline.
