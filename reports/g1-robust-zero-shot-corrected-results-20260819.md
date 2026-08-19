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

## Parameter-Matched Fairness And G1 Intervals

The three-member ensemble has 450,906 trainable parameters. The automatically
selected width-248 single model has 460,382 parameters, 2.1% more than the
ensemble. It uses the same training trajectories, epochs, optimizer family, and
evaluation trajectories.

Seed-level paired improvements, averaged over D2 and D3:

| Seed | Ensemble vs mean member | Ensemble vs parameter-matched single |
|---:|---:|---:|
| 7 | 31.0% | 44.3% |
| 17 | 23.4% | 40.5% |
| 27 | 18.0% | 24.8% |

Paired seed bootstrap (20,000 resamples; only three seeds, therefore G1-level
evidence):

| Metric | Mean | 95% bootstrap interval |
|---|---:|---:|
| Prediction vs mean member | 24.1% | [18.0%, 31.0%] |
| Prediction vs parameter-matched single | 36.6% | [24.8%, 44.3%] |
| Guarded MPC worst-domain distance (initial 3 targets) | 16.7% | [2.9%, 33.9%] |

The two prediction intervals exclude zero under the current seed set. The
initial control interval is superseded by the broader five-target audit below.
This three-seed result is superseded by the five-seed audit below.

## Five-Target Control Audit

Two validation targets were added without retraining or changing the fixed 0.85
guard threshold. Across five held-out targets:

| Seed | Mean worst-domain improvement |
|---:|---:|
| 7 | 7.3% |
| 17 | 30.6% |
| 27 | -0.06% |

The guarded controller improves 11/15 seed-target comparisons and succeeds in
15/15 episodes under the 50 mm tolerance. The seed-level mean improvement is
12.6%, but its three-seed bootstrap interval is [-0.06%, 30.6%] and therefore
crosses zero. This passes the G1 minimum gate (2/3 seeds improve and frozen
deployment remains safe), but it is not a paper-level claim of stable control
improvement. The prediction mechanism remains the stronger result.

## Five-Seed Parameter-Matched Prediction Audit

Seeds 37 and 47 were added without changing the corrected Push protocol,
training budget, model width selection, or evaluation targets. The ensemble has
450,906 parameters; the width-248 single-model baseline has 460,382 parameters
(2.1% more). Both use the same data, epochs, optimizer family, and test rollouts.

| Seed | Ensemble vs mean member | Ensemble vs parameter-matched single |
|---:|---:|---:|
| 7 | 31.0% | 44.3% |
| 17 | 23.4% | 40.5% |
| 27 | 18.0% | 24.8% |
| 37 | 19.5% | 42.0% |
| 47 | 11.4% | 2.1% |

Seed-level paired bootstrap (50,000 resamples):

| Metric | Mean | 95% bootstrap interval | Positive seeds |
|---|---:|---:|---:|
| Prediction vs mean member | 20.7% | [15.3%, 26.4%] | 5/5 |
| Prediction vs parameter-matched single | 30.7% | [15.1%, 42.6%] | 5/5 |

Against the parameter-matched baseline, the mean improvements are 30.0% in D2
and 31.4% in D3, with a positive direction for every seed in both domains.
Seed 47 is weak (0.4% in D2 and 3.7% in D3), so the result supports a stable
direction but also shows substantial initialization variance. This is the
current strongest G1 result; it does not by itself establish a novel ICRA-level
world-model contribution.

Wall-clock recording was enabled for the two added seeds. Ensemble training took
176.7 s and 177.5 s; the parameter-matched model took 54.2 s and 64.3 s on the
same local CUDA device. Earlier checkpoints predate timing instrumentation and
are intentionally reported as missing rather than estimated.
