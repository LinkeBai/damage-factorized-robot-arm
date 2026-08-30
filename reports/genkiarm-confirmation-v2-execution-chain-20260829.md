# GenkiArm confirmation V2 execution chain

## Corrected dependency graph

The five-seed SI-IPWM result cannot be produced from a base world-model
checkpoint alone. The frozen per-seed dependency chain is:

`base -> zero-topology -> contact residual -> physical-context SI-IPWM`

with two additional dependencies:

- `base + zero-topology -> matched adapter`;
- calibrated GenkiArm trajectories -> physical-context encoder.

Only the final physical-context checkpoint, matched adapter and context encoder
may enter the selective-rollout evaluator.

## Frozen implementation

The six training/evaluation configurations are recorded in
`config/experiment/g2_ipwm_genkiarm_*_v2.yaml`. The resumable runner is
`scripts/run_genkiarm_confirmation_v2.py`. It accepts only the preregistered
seeds 107/117/127/137/147, passes `sim/assets/genkiarm_push.xml` to every stage,
uses query seed `training seed + 1000`, and validates each manifest before
skipping a completed stage.

The runner never treats a smoke artifact as complete. Evaluation requires the
physical-context model, both adapter weights, the context encoder, aggregate
metrics and raw rows. A dry run and five pipeline/configuration tests pass.

## Compute environment audit

The original project environment contains CPU-only PyTorch 2.13.0. An isolated
`.venv-cuda` was created with official PyTorch 2.11.0 CUDA 12.8 binaries. It
detects the RTX 4060 and passes the same 16 mechanism/model-contract tests.
CPU and CUDA seed-107 smoke runs match at every printed loss and final metric.
The incomplete CPU formal run created no checkpoint and was restarted from the
same immutable trajectory cache in CUDA; this changes execution hardware only,
not data, seed, model, hyperparameters or gate.

## Evidence boundary

No stage-level legacy gate is silently promoted to the paper result. All
stage-level Pass/No-Go outputs are retained. The five-seed claim is decided
only after all raw final-evaluation rows are aggregated under the V2 evidence
contract.
