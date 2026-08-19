# G1 Robust Zero-Shot Corrected Results (2026-08-19)

## Protocol Correction

The previous Push target split produced zero contact and zero block displacement
for D2 and D3. The tool collision model represented only the TCP capsule and
omitted the lower gripper finger. The task targets were also on the opposite side
of the block from the executable push direction.

Corrections:

- added a measured-scale lower-finger capsule (`pusher_geom`) to `arm_push.xml`;
- separated task goal from the IK control waypoint using a fixed +30 mm x offset;
- replaced the target split with disjoint left-push goals;
- required nonzero contact and displacement in D2 and D3 before training.

All three corrected evaluation targets pass the coverage gate. D2 displacement
is 56.7-67.9 mm with 5-6 contact steps. D3 displacement is 18.3-19.3 mm with
4-5 contact steps. Results from the old collision/target protocol must not be
used as paper evidence.

## Zero-Shot Ensemble Prediction

Three independently initialized topology-conditioned world models were trained
per seed. Values are multi-step RMSE improvement of the ensemble mean relative
to the mean individual member.

| Seed | D2 improvement | D3 improvement | D2 stratified Spearman | D3 stratified Spearman |
|---:|---:|---:|---:|---:|
| 7 | 27.4% | 34.7% | 0.310 | 0.662 |
| 17 | 21.5% | 25.3% | 0.396 | 0.788 |
| 27 | 18.4% | 17.7% | 0.379 | 0.725 |

The prediction direction is consistent in D2 and D3 for all 3 seeds. Ensemble
disagreement remains positively associated with error after stratifying by
rollout depth. The models' aleatoric log-standard-deviation remains unreliable
and is not used for control.

## Guarded Ensemble MPC

Pure ensemble-mean MPC improves D2 but often cancels contact in D3. Minimax and
positive worst-case penalties are over-conservative. The fixed guarded policy
uses ensemble-mean MPC only when its action is within 0.85 L2 distance of the
validated nominal IK action; otherwise it falls back to nominal IK.

Mean worst-domain final block distance over three held-out targets:

| Seed | Nominal IK | Guarded ensemble MPC | Improvement |
|---:|---:|---:|---:|
| 7 | 26.79 mm | 23.24 mm | 13.3% |
| 17 | 26.79 mm | 17.72 mm | 33.9% |
| 27 | 26.79 mm | 26.01 mm | 2.9% |

All nominal and guarded episodes remain below the 50 mm success tolerance.
Eight of nine seed-target comparisons improve; one seed-27 target regresses by
7.1%. The seed-level direction is positive for 3/3 seeds.

## G1 Pivot Decision

The corrected robust zero-shot pivot passes its minimum mechanism gate:

- D2/D3 prediction gains are direction-consistent for 3/3 seeds;
- the frozen guarded controller improves mean worst-domain distance for 3/3 seeds;
- deployment does not update the world models or topology context;
- corrected evaluation trajectories contain real contact and block motion.

This is a G1 mechanism result, not yet a paper-level statistical conclusion.
The next phase should add confidence intervals, parameter-matched ensemble
baselines, compute/parameter accounting, and broader target/domain evaluation.
