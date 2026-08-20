# Fixed-Transform Graph World Model: Gates K0 and K1

**Date:** 2026-08-20
**Decision:** K0 PASS; K1 two-seed provisional PASS; K2 NO-GO

## Motivation

RC-GWM removed a locked joint from the dynamics graph and replaced the two
adjacent links with a generic edge. That reduction is physically wrong: a
locked revolute joint becomes a fixed rigid transform and its link remains in
the kinematic chain. FT-GWM tests the narrower hypothesis that retaining the
complete chain and its fixed SE(3) geometry can enforce the lock without the
free-arm regression observed in RC-GWM.

## K0: exact fixed-transform composition

`fixed_transform_kinematics.py` implements the five-joint chain from
`sim/assets/arm_push.xml` and folds locked rotations and link offsets into
fixed SE(3) transforms. Across D2, D3 and D4, 100 random poses per condition,
the contracted and full chains match each other and MuJoCo's end-effector pose
to machine precision. The focused K0 suite passes (`12 passed` in the complete
kinematics regression run).

Decision: **PASS**. The geometric operator is correct.

## K1: free-joint dynamics

Frozen protocol:

- train on intact, D2 and D4; evaluate held-out D3;
- seeds 7 and 17, 60 epochs, learning rate `1e-3`;
- 150-step trajectories with bounded low-pass goal exploration `0.08`;
- hidden size 128 and 10-step rollout evaluation;
- joint-only loss; no object prediction head;
- pass iff D3 constraint violation is at most `1e-7` and free-arm RMSE
  regression versus the matched graph is at most 5%.

Primary `D3__mixed_composition` results:

| Seed | Matched graph free RMSE | FT-GWM free RMSE | Relative regression | Violation | Decision |
|---:|---:|---:|---:|---:|---|
| 7 | 0.2611 | 0.2701 | +3.45% | 0 | PASS |
| 17 | 0.3891 | 0.2770 | -28.81% | 0 | PASS |

Additional boundaries:

- seed 7 FT-GWM free RMSE on D2/D4/D3-unseen is
  `0.2515/0.2601/0.3490` versus matched graph
  `0.2837/0.2947/0.3476`;
- seed 17 FT-GWM free RMSE on D2/D4/D3-unseen is
  `0.3650/0.2890/0.3295` versus matched graph
  `0.3684/0.3735/0.5143`;
- exact projection produces zero measured lock violation in every evaluated
  domain and seed;
- FT-GWM has 267,650 parameters versus 299,782 for the matched graph. It is
  parameter-favorable, but its explicit per-edge SE(3) implementation is not
  compute-matched and trains more slowly.

Decision: **two-seed provisional PASS**. K1 establishes feasibility of the
fixed-transform representation under the pre-registered fidelity gate. It
does not establish a statistically stable prediction advantage: the matched
baseline itself varies substantially between the two seeds, and seed 7 is
close to the 5% boundary.

## K2: isolated object/contact head

K2 added a 2,340-parameter bottleneck-16 object residual head. It receives the
current object state plus detached pooled joint hidden state and detached
end-effector SE(3) features. An autograd regression test confirms that a pure
object loss gives every joint-transition parameter exactly zero gradient.

K2 v1 incorrectly averaged joint and object dimensions together, diluting the
frozen K1 joint gradient by `10/14`; its result is invalid for architecture
attribution. K2 v2 corrected the objective to `L_joint + L_object`, preserving
the exact K1 joint-loss scale.

Seed-7 primary D3 v2 results:

| Method | Overall RMSE | Free-arm RMSE | Object RMSE | Violation |
|---|---:|---:|---:|---:|
| matched graph | 0.1716 | 0.2212 | 0.0104 | 0.1012 |
| matched graph + projection | 0.1692 | 0.2237 | 0.0108 | 0 |
| FT-GWM K2 | 0.2130 | 0.2701 | 0.1133 | 0 |

FT-GWM's free-arm value is identical to its K1 seed-7 value (`0.2701`), so the
isolation mechanism works. It nevertheless regresses 22.11% relative to the
object-trained matched graph, while object RMSE regresses 986.08%. The detached
low-capacity head cannot infer contact dynamics from a joint representation
that never observes the object; meanwhile the matched graph benefits from
joint/object conditioning and shared training.

Decision: **NO-GO**. Stop the FT-GWM object-prediction branch as registered.
Do not rescue it with post-hoc capacity, contact features, loss weights or
extra epochs. K1 remains a valid constraint-preserving joint-dynamics result,
not a complete Push world model. The stable paper/deployment fallback remains
ensemble uncertainty and selective prediction.

## Artifacts

- `src/robotarm/models/fixed_transform_kinematics.py`
- `src/robotarm/models/fixed_transform_graph.py`
- `scripts/run_ftgwm_gate_k1.py`
- `config/experiment/g2_ftgwm_gate_k1_v1.yaml`
- `config/experiment/g2_ftgwm_gate_k2_v2.yaml`
- `runs/g2_ftgwm_gate_k1/seed7_v1/summary.json`
- `runs/g2_ftgwm_gate_k1/seed17_v1/summary.json`
- `runs/g2_ftgwm_gate_k2/seed7_v2/summary.json`
- `tests/test_fixed_transform_kinematics.py`
- `tests/test_fixed_transform_graph.py`
