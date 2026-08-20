# G2 Gate E: Constraint-Reaction Graph World Model

**Date:** 2026-08-20
**Decision:** PROVISIONAL PASS -- mechanism and five-seed statistics pass;
parameter-matched attribution remains required.

## Hypothesis

A frozen shared graph dynamics model can be adapted to a diagnosed joint lock
by propagating its predicted constraint residual as a learned reaction along
the kinematic chain. The reaction adapter may correct only free joints and the
object; locked position and velocity are enforced analytically.

The leave-one-joint-out protocol trains on intact, D2 and D4 and holds D3 out
entirely. The primary test is `D3__mixed_composition`.

## Primary Results

| Seed | Object RMSE improvement | Free-arm improvement | Overall improvement | Constraint violation |
|---:|---:|---:|---:|---:|
| 7 | +59.85% | -1.50% | +9.98% | 0 |
| 17 | +56.75% | +8.39% | +18.25% | 0 |
| 27 | +22.91% | +13.03% | +22.22% | 0 |
| 37 | +61.26% | +0.28% | +11.69% | 0 |
| 47 | +5.90% | +7.50% | +17.23% | 0 |

Seed-level bootstrap with 100,000 resamples:

| Metric | Mean improvement | 95% CI | Direction |
|---|---:|---:|---:|
| Object RMSE | **+41.33%** | **[+20.09%, +59.79%]** | 5/5 positive |
| Free-arm RMSE | **+5.54%** | **[+0.84%, +10.07%]** | 4/5 positive |
| Overall RMSE | **+15.87%** | **[+11.97%, +19.63%]** | 5/5 positive |

The harder `D3__mixed_unseen` condition retains the same qualitative behavior
in all five runs. The adapter also enforces zero lock-position/velocity
violation in every evaluated domain.

## Attribution Audit

Earlier gates established the following chain:

1. hard output projection alone is unstable across seeds;
2. a shared chain graph strongly outperforms the dense RSSM, but a same-graph
   ablation shows that graph architecture explains most of that difference;
3. topology surgery without reaction propagation does not stably improve both
   free-arm and object prediction;
4. the constraint-reaction adapter is the first topology-aware method whose
   five-seed seed-level confidence intervals are positive for all three primary
   metrics.

## Remaining Fairness Blocker

`graph_ordinary` has 169,542 parameters. The full frozen-base plus reaction
model contains 291,373 parameters. Although only the reaction adapter is
trainable in Stage 2, the deployed model is larger. Therefore this gate does
not yet establish that the improvement is uniquely caused by the proposed
reaction structure.

Before promoting the result to a final method claim, compare against:

- a graph-ordinary model matched to total deployed parameters;
- a frozen graph base plus an equally sized unconstrained residual adapter;
- a wrong-lock and shuffled-lock reaction ablation;
- reaction message-step ablations and D2/D4-to-D3 leave-one-joint-out repeats.

## Artifacts

- `config/experiment/g2_constraint_reaction_gate_e_v1.yaml`
- `config/splits/g2_topology_leave_one_joint_out_v1.yaml`
- `src/robotarm/models/constraint_reaction_world_model.py`
- `src/robotarm/models/topology_graph_world_model.py`
- `runs/g2_constraint_reaction_gate/seed{7,17,27,37,47}_v1/`
