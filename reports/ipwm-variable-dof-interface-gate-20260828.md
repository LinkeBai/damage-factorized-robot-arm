# IPWM variable-DoF interface Gate

**Decision:** INTERFACE PASS; CROSS-ROBOT PREDICTION UNPROVEN

## Audit finding

The current published implementation is not variable-DoF.  The fixed-transform
model stores five GenkiArm axes/origins, enumerates four fixed edges, embeds a
14-D state layout and uses GenkiArm-specific pusher geometry.  The physical
context encoder defaults to state/action/topology dimensions 14/5/5; training
losses, planner geometry and protocol loader also contain five-joint slices.
Changing a single `dof` field or padding Panda into the old tensor would not be
a valid cross-structure experiment.

## Minimal interface result

`VariableDofInterventionCore` preserves the original mechanism rather than
adding a new prediction component:

1. each valid revolute joint is a graph node with local q/qvel/action, lock
   indicator, normalized chain depth and MJCF axis/origin;
2. one node encoder, edge function, update and head are shared at every joint;
3. the analytic intervention projects locked q to its declared angle, locked
   qvel/action to zero, before and after learned prediction;
4. an explicit valid-node mask makes padding a batching implementation only.

The same instantiated parameter set accepts all five GenkiArm joints and all
seven Panda joints.  No robot ID, per-robot head or joint deletion is used.
For a 5-node sample evaluated alone and inside a 7-wide padded tensor, valid
predictions agree within absolute tolerance `1e-7`; invalid outputs are zero.
Locked position equals the supplied floating-point lock angle exactly and
locked velocity is exactly zero.

## What this does not prove

The module currently covers the structural robot transition only.  It is not
connected to the original object/contact head, context encoder, planner or
checkpoint and has not been trained.  Therefore this Gate does not establish
prediction accuracy, few-shot adaptation, object propagation, Push/Grasp,
action ranking, control, or cross-robot generalization.  It adds zero points to
the paper score.  The next Gate must define a common variable-size trajectory
contract and compare shared-mechanism versus per-robot/unstructured baselines
without using robot identity or privileged forces.

## Unified trajectory and actuator contract

The follow-up interface now extracts both real MJCFs into one lossless schema:

- variable-length full joint nodes `[q, qvel]`, normalized commanded/applied
  actions, lock mask and lock angles;
- object world pose `[xyz, wxyz]` and twist `[linear xyz, angular xyz]`;
- contact mask and optional gripper state;
- explicit node/time validity masks during batching, with no robot identity
  field in the tensors consumed by a model.

GenkiArm's planar block is embedded with its simulator-constant z/orientation
and zero unsupported twist components; Panda retains its full free-joint state.
The contract rejects non-unit quaternions, non-finite data, truncated joint
sets and any locked trajectory whose q/qvel/applied action violates projection.

`VariableMujocoArmEnv` exposes the same normalized action interface for both
robots while documenting the low-level boundary: direct-force actuators receive
normalized force, and official affine position actuators receive a bounded
target increment.  This mapping stays outside the learned model.  Five-step
tests confirm the same analytic pinning rule holds exactly for GenkiArm and
Panda locks.  This completes the data/interface prerequisite, not the
cross-robot prediction Gate.
