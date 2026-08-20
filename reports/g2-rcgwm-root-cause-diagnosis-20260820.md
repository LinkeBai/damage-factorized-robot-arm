# RC-GWM Root-Cause Diagnosis

**Date:** 2026-08-20
**Scope:** seeds 7 (successful initialization) and 17 (failed initialization)

## Bottom line

RC-GWM's exact constraint operator is correct. Its prediction failure is
primarily an optimization multi-stability problem on a deterministic, very
low-diversity training set, amplified by shared free-arm/object recurrence.
Several plausible graph-representation explanations were experimentally ruled
out. Lowering the learning rate substantially stabilizes training but does not
fully recover held-out-topology free-arm fidelity, so the current architecture
still lacks a reliable inductive bias for unseen lock locations.

## J1: final checkpoint versus validation checkpoint

Intervention: unchanged RC-GWM; record per-epoch losses and shared-gradient
statistics; select the best checkpoint on the existing D2/D4/intact validation
domains.

- seed 7: best epoch 19; primary free-arm `0.2030` versus final `0.2095`.
- seed 17: best epoch 18; primary free-arm `0.2253` versus final `0.3882`.
- negative free/object gradient epochs: 25% for seed 7, 50% for seed 17.

Decision: late/path instability is causal and large for failed seeds, but the
best seed-17 checkpoint still regresses about 6.8% versus matched graph and its
object RMSE worsens. Existing validation topologies also rank D3 checkpoints
poorly. Checkpoint selection is necessary but insufficient.

## J2: stop object gradients into the shared joint graph

Intervention: detach both pooled hidden features and recurrent object inputs
from the joint graph; object head remains trainable.

- seed 7 final primary free/object: `0.1933/0.0223` versus J1 `0.2095/0.0153`.
- seed 17 final primary free/object: `0.3097/0.0455` versus J1 `0.3882/0.0645`.

Decision: shared free-arm/object recurrence contributes causally to the
collapse. It is not sufficient: seed 17 remains far above the matched free-arm
baseline `0.2110`.

## J3: explicit bridge-edge type, span and direction

Intervention: add bridge indicator, original-chain span and direction to each
contracted edge.

- seed 7 final primary free/object: `0.2230/0.0188`.
- seed 17 final primary free/object: `0.2815/0.0668`.

Decision: simple missing edge-type information is not the main cause.

## J4: true packed active-node graph

Intervention: pack free joints into contiguous reduced-coordinate slots so
locked nodes never enter encoder, message passing or GRU computation.

Result: every epoch loss, gradient statistic and test metric is numerically
identical to J1 for both seeds.

Decision: fixed-slot masking and packed reduction are functionally equivalent
under shared, permutation-equivariant node/message/GRU parameters. The masked
implementation is not the cause.

## J5a: double trajectories per domain

Intervention: increase `trajectories_per_train_domain` from 2 to 4.

- seed 17 final primary free/object: `0.3870/0.0642`, essentially unchanged
  from `0.3882/0.0645`.
- direct tensor audit: seed-7 and seed-17 `goal` trajectories are exactly
  identical. The goal collector does not use its random seed.
- additional trajectories cycle over the finite target list, so this change
  does not add independent excitation.

Decision: the nominal five data seeds are actually initialization seeds. The
training set is deterministic and low-diversity; simply increasing its count
does not add information.

## J1b: lower learning rate at matched cumulative budget

Intervention: seed 17, learning rate `1e-3`, 60 epochs instead of `3e-3`, 20
epochs.

- validation becomes smooth rather than sharply non-monotonic.
- primary free/object improves from `0.3882/0.0645` to `0.2428/0.0086`.
- matched graph free-arm remains better at `0.2110`.

Decision: the original step size is a major cause of multi-stability. Lower
learning rate fixes object prediction and much of free-arm collapse, but does
not establish held-out-topology free-arm non-regression.

## J3b: retain the contracted joint's fixed rotation

Intervention: add `sin(lock_angle)` and `cos(lock_angle)` to the contracted
edge. D2/D3/D4 use `+0.5/-0.5/+0.9` rad respectively.

- seed 7 final primary free/object: `0.2309/0.0401`.
- seed 17 final primary free/object: `0.3172/0.0268`.

Decision: deleting a locked joint without retaining its fixed transform is a
physical modeling defect, but the tested scalar-angle representation does not
resolve the instability. A correct future contraction would need full
kinematic transform composition rather than generic scalar edge features.

## Root-cause ranking

1. **Confirmed major:** learning-rate/rollout optimization multi-stability.
2. **Confirmed contributor:** free-arm/object shared-gradient and recurrent
   coupling.
3. **Confirmed protocol weakness:** deterministic low-diversity goal data;
   reported seeds vary initialization, not trajectories.
4. **Confirmed modeling defect, insufficient fix:** contraction drops the
   locked joint's fixed kinematic transform.
5. **Ruled out:** fixed-slot masked graph versus packed active graph.
6. **Ruled out as simple fix:** generic bridge type/span/direction features.

## Recommendation

Do not run more RC-GWM variants on the existing deterministic data. The next
scientifically valid gate must first create genuinely diverse goal trajectories
(bounded exploration around the controller) and then compare a stable
optimizer plus decoupled joint/object recurrence. If architecture work resumes,
contract edges with URDF-derived SE(3) transforms, not learned scalar bridge
labels. Until that protocol exists, ensemble uncertainty/selective prediction
remains the stable paper route.

## J6: Diverse goal exploration plus stable optimization

Adding bounded low-pass Gaussian exploration (`std=0.08`) made seed 7 and seed
17 training trajectories genuinely different while preserving contact and
block displacement. With the stabilized `lr=1e-3 / 60 epoch` schedule, both
runs trained smoothly and avoided catastrophic final-checkpoint divergence.

Primary free/object RMSE was `0.2436/0.0087` for seed 7 and `0.2448/0.0085`
for seed 17. Matched-graph free-arm references were approximately `0.2121`
and `0.2110`, so RC-GWM still regressed by roughly 15% on free-arm prediction
despite excellent object prediction and exact zero violation.

This separates the causes: data diversity and optimizer stability fix the
multi-stability, but the current reduced-coordinate inductive bias remains
inadequate for free-arm held-out-topology transfer.

## Final diagnosis

Do not spend more runs adding generic edge features, packed slots, or
unregistered loss weights. A future architecture must preserve the physical
transform of a locked link and separately model free-joint dynamics; otherwise
the stable ensemble uncertainty route is preferable.

## Artifacts

- `scripts/diagnose_rcgwm_optimization.py`
- `config/experiment/g2_rcgwm_diagnosis_j{1,2,3,4,5a}_v1.yaml`
- `config/experiment/g2_rcgwm_diagnosis_j{1b,3b}_v1.yaml`
- `runs/g2_rcgwm_diagnosis_j*/seed*_v1/summary.json`
