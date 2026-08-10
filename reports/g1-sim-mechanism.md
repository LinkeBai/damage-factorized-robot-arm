# G1 Simulation Gate Review

Date: 2026-08-10

Status: **COMPLETE - NO-GO / PIVOT REQUIRED**

## Scope

- Calibrated five-joint MuJoCo model after G0.
- Reach task, D2 and D3 locked-joint deployments.
- Three seeds: 7, 17, 27.
- Calibration shots K = 0, 1, 2, 5.
- topology-only, residual-only, monolithic matched, and factorized DFWM.
- Calibration and evaluation targets and trajectory seeds are disjoint.
- Deployment updates only the 8-D residual code; model and planner weights stay frozen.

## Prediction Evidence

The final fixed configuration used 50-step trajectories, 40 training epochs,
and 20 latent-optimization steps. A 100-step/50-latent-step configuration was
terminated after the 40-minute execution limit and produced no result artifact.

| Model | K | Held-out state RMSE |
|---|---:|---:|
| topology-only | 0 | 0.07042 +/- 0.01648 |
| DFWM | 0 | 0.07649 +/- 0.01183 |
| DFWM | 1 | 0.06798 +/- 0.01454 |
| DFWM | 2 | 0.06763 +/- 0.01426 |
| DFWM | 5 | **0.06760 +/- 0.01391** |
| residual-only | 5 | 0.08728 +/- 0.01851 |
| monolithic matched | 5 | 0.06950 +/- 0.01596 |

K=5 improves mean RMSE by 4.0% over topology-only. Seeds 17 and 27 improve
on both D2 and D3; seed 7 degrades on both. Prediction therefore meets the
minimum 2/3-seed direction test, but the effect is small and seed-sensitive.
Increasing K improves DFWM from K=0 to K=1 and then largely plateaus.

## Frozen MPC Evidence

The original 25-step smoke represented only 0.125 seconds of MuJoCo time and
was invalid as a Reach control test. After correction, evaluation used 400
steps (2 seconds), 100-step training trajectories, and frozen CEM MPC.

| Method | Success | Mean final distance |
|---|---:|---:|
| topology-only | 0/2 | 0.6687 m |
| DFWM K=0 | 0/2 | **0.3000 m** |
| DFWM K=1 | 0/2 | 0.5123 m |
| DFWM K=2 | 0/2 | 0.5119 m |
| DFWM K=5 | 0/2 | 0.5125 m |

Residual-state prediction removed autonomous rollout divergence, but no method
reached a target. Few-shot adaptation also failed to improve frozen-MPC control
monotonically. The required frozen-control benefit is absent.

## Gate Decision

G1 is complete but does **not** pass the Go gate. Do not start G2 or claim
few-shot recovery. The evidence supports only a narrow, simulation-only
prediction result.

Pivot actions:

1. Freeze the current result as a prediction benchmark.
2. Build a healthy-arm Reach controller that succeeds before adaptation.
3. Collect controller-induced trajectories rather than short random torque data.
4. Add multi-step rollout loss and validate open-loop horizon error.
5. Re-open the control gate only after topology-only succeeds reliably and
   factorized K>0 improves normalized return in at least two seeds.

The 2026-08-10 pivot recheck established a 4/4 deterministic controller for
intact, D2, and D3, then reran frozen MPC with controller-induced data and a
five-step rollout loss. Control success appeared only in seed 7, not seeds 17
or 27. The formal decision therefore remains No-Go; see
`reports/g1-pivot-control-baseline.md`.

## Artifacts

Versioned evidence is under `results/final/g1-benchmark-20260810/`.
The formal run used 846.5 seconds wall-clock, 0.235 GPU-hours, and 76.3 MB peak
allocated GPU memory. All repository tests pass after the model correction.
