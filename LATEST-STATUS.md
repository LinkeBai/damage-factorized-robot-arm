# Latest project status

Last updated: 2026-08-30

## Current empirical decision

The primary platform is the original physical 5-DoF arm and its
`sim/assets/arm_push.xml` kinematic task model. GenkiArm and Panda are
transfer platforms; they cannot replace a missing primary-platform result.
The XML has not yet been identified as a full dynamic digital twin.

The latest original-arm carrier screen gives:

| Fault | Hard mask | One-step SFET transport | Fault-aware constrained IK |
|---|---:|---:|---:|
| D2, seeds 7/17/27 | 20/0/20% | 20/0/20% | 100/100/100% |
| D3, seeds 7/17/27 | 0/0/0% | 0/0/0% | 80/100/100% |

The method previously labelled `oracle_ik` uses only the known lock,
target and public kinematics. It is therefore a deployable strong baseline,
not an oracle. One-step task-effect transport does not create a feasible
contact path and is a task-level No-Go.

The calibrated GenkiArm confirmation remains a transfer boundary: routed
selective SI-IPWM changes object RMSE by +0.6037%, -0.7836% and +0.7182% for
seeds 107/117/127 (mean +0.1794%; interval crosses zero), while preserving
zero free-state regression and zero lock violation.

> **Only hard-lock feasibility and carrier-relative state isolation are
> currently supported. Stable task-performance superiority is not.**

The current objective ICRA/CCFA assessment remains 3.2--3.4/5. No-Go results
improve auditability, not the scientific score.

## What recent work established

- Corrected arbitrary-direction Push waypoints and verified the original-arm
  control/data interface.
- Reproduced the upstream TD-MPC2 short training path, transferred its
  interface to the original arm, and diagnosed zero-contact random coverage.
- Ran healthy DAgger, fault D3 five-trajectory DAgger, BC and HCAR few-shot
  baselines; these establish learnability and strong comparison requirements,
  but no stable IPWM advantage.
- Reproduced the released LeWM PushT checkpoint-planner-control path at smoke
  scale; this is a pipeline reproduction, not the paper's aggregate result.
- Audited DINO-WM, DyWA, DreamFLEX, multi-joint-failure NPM and related
  artifacts, recording code/data/runtime blockers without inventing scores.
- Implemented SFET and its structural tests. Its synthetic 3-shot signal does
  not transfer to the current robot task, so it is not a paper result.
- Preserved corrected action-ranking, bilinear/secant upper-bound and
  provenance diagnostics as No-Go evidence.

## Evidence boundaries

- Supported: exact locked-coordinate projection, carrier-relative
  non-interference/state isolation, reproducible diagnostic pipelines.
- Not supported: stable object prediction or action-ranking superiority on the
  primary arm, closed-loop advantage, Panda object/contact transfer, method
  Grasp, visual closed loop or real-robot benefit.
- Panda scripted Grasp 5/5 is task feasibility only.
- Dual fixed eye-to-hand visibility is camera-layout feasibility only.
- Large local `runs/`, external repositories, checkpoints and downloaded
  papers are not part of the public Git repository.

## Frozen next direction

All compared methods first share the fault-aware constrained-IK carrier. The
only remaining core question is whether three paired fault trajectories can
use the frozen pre-fault IPWM as a control variate to improve unknown
contact-effect margins, action ranking, top-1 regret and terminal error under
held-out targets and physics. The candidate name in the plan is
FCCM-IPWM; it is a preregistered Stage-0 hypothesis, not a successful method.

Stage 0 must beat same-data direct ridge/physics identification and history or
FiLM baselines, improve closed-loop metrics, and lose its gain when the IPWM
prior or intact/fault pairing is removed. Otherwise the IPWM attribution is
No-Go and the claim is reduced rather than repackaged.

The original 5-DoF real experiment on September 1 prioritizes synchronized raw
trajectories, dual eye-to-hand video, lock/safety truth, three fixed
calibration trajectories and a blind candidate-action library. Cross-arm Push
and Grasp are only expanded after the original-arm core Gate.

See:

- `PROJECT-PLAN-V6.md`
- `reports/to-senior-2100-progress-20260830.md`
- `reports/reproduction-first-audit-20260830.md`
- `reports/sfet-task-level-nogo-20260830.md`
- `reports/icra-16-day-primary-platform-reset-20260830.md`
