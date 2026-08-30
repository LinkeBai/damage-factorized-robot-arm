# Paper Claim--Evidence Audit (2026-08-29)

## Decision

The manuscript does not falsely claim successful closed-loop control,
cross-robot generalization, Grasp/Pick, or real-robot validation. Its present
claim is a narrow, simulation-only SI-IPWM result for five-joint planar Push:
analytic lock feasibility plus state-isolated object correction relative to a
trusted carrier.

That narrow claim is internally supported, but the manuscript does **not** meet
the frozen 4.0/5 ICRA evidence contract. The current objective score remains
**3.2--3.4/5 (weak reject)**. Honest negative results improve integrity, not
task-level evidence or significance.

## Claim--artifact ledger

| Manuscript claim | Direct evidence | Status / action |
|---|---|---|
| Locked position/velocity are satisfied exactly | analytic projection, unit tests, zero reported violation | Supported in the evaluated simulator; retain, but not as stand-alone strong novelty |
| Published free-joint and pusher states equal the carrier | dual-private-rollout proposition and 27/27 audit cells | Supported; equality is relative to the carrier, not truth |
| Object RMSE improves in routed physical-OOD regimes | frozen seeds 27/37/47 and post-design query seed 57 | Supported but limited; query seed 57 is not a new training seed |
| Prediction improvement yields useful control | guarded MPC improves only 1/3 seeds | **No-Go**; prohibit control-superiority language |
| IPWM transfers to calibrated GenkiArm | `g2-ipwm-genkiarm-zero-shot-transfer-20260828.md` | Partial only; do not claim deployment/generalization |
| One shared model handles variable 5/7 DoF | variable-DoF interface tests | Interface only, not performance evidence |
| Shared structure improves held-out-lock robot transition on two arms | cross-arm prediction Gate, 2/3 seeds | Narrow provisional result; both structures were in training |
| Fault influence propagates through contact to the object across arms | cross-arm contact Gate, 0/3 seeds | **No-Go**; exclude from paper claims |
| Solver-native response works on a held-out lock | deployable response Gate, worse on 2/3 seeds | **No-Go**; freeze route |
| Push and Grasp/Pick are validated | no complete dual-task result | Unsupported; do not claim |
| Two-arm full task generalization | wrappers exist but object/contact Gate failed | Unsupported; do not claim |
| Dual eye-to-hand visual robustness | camera geometry only, no complete visual chain | Unsupported; do not claim |
| Real-robot or sim-to-real validity | no hardware experiment | Unsupported; retain simulation-only limitation |

## Criterion score

| Criterion | Score / 5 | Reason |
|---|---:|---|
| Novelty | 3.3--3.5 | State isolation is clear, but the stronger propagation/operator hypothesis failed its deployable gates. |
| Soundness | 3.7--3.9 | Narrow invariants are careful; post-hoc design and three inspected checkpoints limit inference. |
| Evidence | 2.9--3.2 | Only single-arm planar Push has a complete positive chain; control, cross-arm contact, Grasp, five untouched seeds, and hardware are absent or No-Go. |
| Significance | 3.0--3.3 | Safety isolation is useful, but no stable task benefit is demonstrated. |
| Clarity | 3.8--4.1 | Scope and limitations are explicit, though the failure ledger competes with the central story. |
| Reproducibility | 4.0--4.2 | Frozen configs, raw-row references, tests, and retained failed gates are strong. |
| Overall | **3.2--3.4** | Below 4.0 because the decisive task/external-validity requirement is unmet. |

## Frozen writing boundary

Until a new mechanism passes its preregistered Gate, the paper may claim only:

> Under diagnosed single-joint locks in one simulated planar Push setting,
> SI-IPWM analytically preserves lock feasibility and prevents an intervention
> object correction from changing the carrier's published robot/pusher state;
> it improves routed object prediction in the audited cells but does not yield
> consistent closed-loop improvement.

Variable-DoF compatibility, the 2/3 robot-transition Gate, the 0/3
object/contact Gate, the failed solver-response Gate, planned Push/Grasp runs,
and planned visual/hardware validation must not be merged into an implied
successful system.

## Route to 4.0+

Repackaging cannot achieve it. Before a five-seed matrix, one new falsifiable
core hypothesis must pass a small frozen development Gate connecting diagnosed
fault structure to **object/contact prediction and action ranking**, not merely
robot-state prediction. It must beat an identical-observable,
parameter-matched flat baseline on both robot structures, show positive
direction in at least 2/3 development seeds without per-seed exceptions, and
improve a preregistered task-relevant ranking metric. Only then is expansion to
Push + Grasp/Pick and five untouched seeds justified.

If no such hypothesis is available, the professional decision is a narrower
prediction/safety paper at a better-matched venue or waiting for hardware, not
assigning an ICRA score above the evidence.
