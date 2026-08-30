# BT-DPWM Y6 Three-Seed Replication

Date: 2026-08-21. Frozen method: Y6 block-coordinate BT-DPWM. Primary test
domain: `D3__mixed_composition`, rollout depth 10.

Each replication seed trains its own 120-epoch `h=96` shared-graph baseline.
BT-DPWM trains the contact-conditioned robot block for 120 epochs at horizon 10,
freezes it, and then trains the independent recurrent object block for 120 epochs
at horizon 5. No checkpoint selection or post-result threshold change is used.

| Seed | Free-arm improvement | Object improvement | Overall improvement | Gate |
|---:|---:|---:|---:|---|
| 7 | +4.16% | +3.69% | +4.16% | PASS |
| 17 | +10.50% | +6.78% | +10.48% | PASS |
| 27 | +9.02% | +5.44% | +8.99% | PASS |
| **Mean** | **+7.90%** | **+5.30%** | **+7.88%** | **3/3 PASS** |
| Sample SD | 3.32 pp | 1.55 pp | 3.30 pp | — |

Every evaluated model has zero analytic constraint violation. These results
upgrade Y6 from a single-seed provisional pass to a three-seed mechanism pass
against the compute-matched shared graph on the frozen primary domain.

## Execution optimization

Deterministic CPU MuJoCo trajectories are now cached by protocol, seed, domain,
target/excitation settings, and collection parameters. Seed 17/27 train and test
datasets were materialized in `runs/trajectory_cache`; reruns avoid collection.

The model exposes a robot-only forward path, but it is disabled in the formal
replication because recursively predicted object/contact state affects later robot
transitions. Enabling it would change Y6 semantics. Frozen robot rollouts are also
not cached across object-training epochs for the same reason. The existing Warp
collector remains excluded from this gate until goal-policy, contact, delay, and
deadband parity are implemented and tested.

## Remaining evidence boundary

This is not yet an ICRA-complete claim. Required next gates are parameter/compute
accounting, multi-domain/topology replication, direct comparisons to public strong
baselines, downstream planning/control, and real-robot validation.
