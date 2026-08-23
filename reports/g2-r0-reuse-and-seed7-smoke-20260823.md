# G2-R0 reuse audit and seed7 decisive smoke (2026-08-23)

## Corrected task definition

`D1`--`D5` denote which single joint is locked, not the number of simultaneous
damages. The frozen task is therefore **held-out topology × held-out residual
physics**, with D3 absent from base training and D2/D4 serving seen-topology
controls. No result in this report is described as multi-joint composition.

## What is reused

| Artifact | R0 treatment | Reason |
|---|---|---|
| MuJoCo train/validation/test trajectories and caches | reuse | data protocol and seeds are unchanged |
| Z32 compute-matched shared h136 | reuse | strongest unchanged base comparator |
| Z69 BT checkpoint | reuse initialization | preserves learned robot/object dynamics |
| analytic topology projection and zeroed topology columns | reuse | already verified by G2 structural ablations |
| Z70 shared/BT adapters and Z65 posterior | defer/reuse in full R0 | K=0 architectural smoke must pass first |
| Z75 support/hysteresis/z0 safety policy | defer/reuse | decision policy is not the current object-model bottleneck |
| G2 robustness/calibration/failure evidence | reuse as frozen baseline | new method has not passed R0, so rerunning is premature |
| shared/robot/object full training | not rerun initially | only the changed block should train |

`ponytail-main` is a development-process plugin, not a dynamics model, dataset,
or checkpoint. Its only applicable guidance is to reuse existing helpers and
leave a focused test; it contributes no scientific mechanism.

## Smoke A: geometry residual on the old forward coupling

The first frozen extension added a 276-parameter, zero-initialized explicit
pusher/object geometry residual and trained only those parameters. Relative to
the frozen Z69 BT checkpoint, D3 H10 object RMSE improved from 0.048389 to
0.043700 (about 9.69%). Relative to shared it improved 1.99%, just below the
frozen 2% smoke threshold. Free RMSE nevertheless regressed 2.54% versus shared.

This exposed a structural contradiction: the G2 checkpoint used
`contact_conditioned_robot: true`. Although object gradients cannot update the
robot block, predicted object state enters the next robot forward step. Thus the
implementation was gradient-directed but not a strict forward block triangle.

## Smoke B: strict forward triangle plus geometry

The second protocol removed object-to-robot forward feedback, copied the six
physical robot-input columns from Z69, retrained only the robot block for 40
epochs with validation selection, and trained the same 276-parameter geometry
residual. D3 H10 results were:

| method | free RMSE | object RMSE | overall RMSE | violation RMSE |
|---|---:|---:|---:|---:|
| shared + projection | 0.206041 | 0.044585 | 0.157567 | 0 |
| strict BT + geometry | 0.191688 | 0.044639 | 0.146856 | 0 |

This is the desired short-horizon pattern: free +6.97% and overall +6.80%
relative improvement, with object differing by only -0.12% and zero violations.
However, the preregistered object-superiority gate remains NO-GO.

## Multi-horizon falsification

The frozen evaluator then measured every test domain at H=1/5/10/25/50 and
included the exact same strict model with its geometry head zeroed. On primary
D3 mixed-composition:

| horizon | free gain vs shared | object gain vs shared | geometry object gain vs no-geometry |
|---:|---:|---:|---:|
| 10 | +6.97% | -0.12% | +7.72% |
| 25 | +11.32% | -50.54% | -16.28% |
| 50 | +24.71% | -174.18% | +2.07% |

Across D2/D3/D4, strict BT's free advantage generally grows with horizon, while
object rollout is unstable and substantially worse than shared at long horizons.
The geometry branch often improves its matched no-geometry ablation (for example
D2 H25 +56.63% and H50 +74.40%), proving a measurable causal contribution, but
it does not close the absolute gap to shared.

## Decision and next work

Do not expand seeds yet. R0 has isolated a real positive mechanism—strict
damage-projected robot dynamics—and a real remaining blocker—long-horizon object
stability. The next change must remain inside the object block: train an
increment/velocity-consistent geometric transition with multi-horizon stability
and selection on held-out validation physics. Shared, trajectory caches, strict
robot checkpoint, projection, and the new evaluator remain reusable. Only the
object block and its matched ablation need rerunning before seed expansion.

## Support-aware intervention residual continuation

The frozen-head diagnostic showed that the shared object head can be reused at
short horizon, but strict-robot latent drift accumulates at H25--H50. Updating
the whole compact head improved D3 while damaging D2/D4, so the reusable shared
head is now frozen and correction is isolated in a zero-initialized residual.
Routing is label-free: the analytic damage mask is compared with the frozen
D1/D2/D4/D5 training support. Seen masks use the base at evaluation; unseen D3
uses the meta-trained residual.

The 276-parameter geometry-only v1 missed the primary threshold (D3 H10 object
+1.24%), although it improved D3 mixed-unseen by +8.33%/+11.23% at H10/H25.
The frozen v2 adds a rank-32 latent residual (4,644 parameters) while retaining
the geometry branch and freezing the 19,724-parameter shared object head. It
selected epoch 34 using only the validation split and passed the primary seed7
gate: object +3.93%, free +2.39%, overall +2.42%, violation 0.

| test domain | H10 object | H25 object | H50 object |
|---|---:|---:|---:|
| D3 mixed composition | +3.93% | +13.56% | +28.02% |
| D3 mixed unseen | +14.09% | +43.26% | +31.68% |
| D2 mixed composition | -2.21% | -4.34% | -22.42% |
| D4 mixed composition | -2.89% | -4.67% | -36.56% |

Residual ablation removes up to 50.80% matched object improvement on D3, while
its contribution is exactly zero on every D2/D4 row. Thus the core held-out
topology mechanism has a decisive seed7 signal without contaminating supported
topologies. The unified gate is nevertheless not passed: the frozen base still
receives a shifted strict-robot latent and loses long-horizon object accuracy on
seen-topology/OOD-physics controls. Seed17/27 remain unopened until a reusable
base-alignment or K-conditioned physics-mismatch route closes that control gap.

## Alignment and stability continuation

A frozen shared-bridge oracle established causality: replacing only the strict
bridge code recovered D2 object gains to -0.19/+0.01/+1.88% and D4 to
-0.42/+0.33/+1.29% at H10/H25/H50. A deployable rank-32 object-side alignment
head then reduced validation bridge MSE by about 94.5% without duplicating the
shared encoder. The selected v3 retained strong D3 object gains while limiting
the worst D2/D4 object regression to 3.87%. Its worst pusher regression was
2.556 mm in absolute terms, inside the frozen 3 mm hardware-facing equivalence
margin. An object-output-weighted v4 traded away D3/D4 performance and was
rejected.

The frozen Z70 adapter/Z65 encoder/Z75 safety chain loaded without retraining,
but accepted only one of sixteen nonzero-budget cells; that D4 update worsened
BT-own object RMSE by 0.81%. Force-enabling the intervention residual on seen
topologies harmed D2/D4 H10/H25 object metrics by roughly 9--24%, so K cannot
simply toggle the same residual.

Seed17 exposed the next blocker before object training: the identical strict
robot protocol achieved D3 H10 free gain -6.58%, outside the preregistered 5%
margin. Epoch0 was -10.91%, and a further 40-epoch continuation selected on the
same validation objective was -7.39%; extra optimization is not the solution.
Seed17 is now development evidence, not untouched confirmation, and seed27 was
not opened. The object mechanism remains frozen; subsequent work is restricted
to leave-one-joint-out intervention-equivariant robot selection, followed by
new untouched confirmation seeds only after development stability.

## Contact-aware base and long-horizon residual audit

The strict-forward robot assumption, rather than object capacity, caused the
cross-seed robot failure. Removing topology inputs worsened seed17 D3 H10 free
gain to -13.71%; analytic contact gating without/with recalibration reached
-10.92%/-6.99%. In contrast, the frozen contact-aware Z69 robot is stable at
-0.13/-0.42/-1.23% on seeds 7/17/27. Contact reaction is therefore physically
necessary in pushing. The method remains block-coordinate and projected, but a
strict forward-block-triangular claim is not supported.

Combining that stable robot with the frozen shared object base and the new
support-aware residual produced the strongest v5 result. Seed7 D2/D4 are
numerically identical to shared at every horizon, while D3 composition object
gains are +9.34/+9.73/+30.88% and mixed-unseen gains are
+17.20/+44.85/+30.31%. Seed17 confirms H10/H25 gains but fails at H50
(-129.05/-108.64%), so seed27 remains untouched.

Fixed scaling, relative clipping, geometry-only rank expansion, and H50-only
training did not remove the seed17 mixed-unseen failure. A depth-decayed
residual (0.85) made all six seed17 D3 cells positive after post-hoc application,
but the same value missed the 2% seed7 H50 threshold. Training with the decay
made seed17 composition H50 +2.03% while mixed-unseen H50 remained -8.55%.
The remaining target is now isolated to held-out topology plus held-out physics
at H50. Future work must condition the bounded residual coefficient on the
already available physical-context/K posterior; further unconditional scale or
clip tuning is not justified.
