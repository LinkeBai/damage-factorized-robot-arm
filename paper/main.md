# Damage-Factorized World Models for Few-Shot Adaptation to Joint-Locking Failures

> 论文方法部分草稿（ICRA 2027）。当前基于仿真预测证据，控制/真机证据待补。
> 正文英文，`<!-- -->` 内为中文待办/说明。

<!-- TODO: 摘要、引言、相关工作、实验、结论等章节待补。本文件先写 Method 核心。 -->

---

## Abstract (draft)

<!-- TODO: 待结果冻结后写。核心主张：因子化离散拓扑与连续残余，部署时 latent-opt 推断 8 维残余，冻结 WM，在 held-out topology-residual 组合上优于所有 baseline。 -->

---

## I. Problem Formulation

We consider a 5-DoF serial manipulator arm (base yaw, shoulder pitch, elbow
pitch, wrist pitch, wrist roll) plus an independent gripper. A deployment
suffers a single-joint **locking failure**: one positioning joint is
immobilized at a fixed angle. We assume:

- **Diagnosis** provides the *identity* and *lock angle* of the failed joint
  (a discrete, cheaply-obtained signal).
- The **residual dynamics** that remain unknown after diagnosis — actuator
  degradation, joint friction/compliance variation, command latency, deadband,
  payload — are continuous and can only be estimated from a small number of
  safe calibration trajectories.
- At most `K ∈ {0,…,5}` low-risk calibration trajectories are available,
  collected with evaluation targets held out.

The goal is to recover predictive control of the damaged arm with minimal
interaction. We formalize the damaged dynamics as a conditional transition
model

    p(o_{t+1} | o_t, a_t, c_damage),

where `o_t ∈ R^{10}` is proprioception (5 joint positions + 5 velocities),
`a_t ∈ R^5` is a normalized joint command, and `c_damage` is a deployment-
specific damage context.

---

## II. Method: Damage-Factorized World Model (DFWM)

The key idea is to **factor** the damage context into two components with
fundamentally different sources and roles:

    c_damage = [ e_topology , z_residual ]

- `e_topology ∈ R^{64}` encodes the **discrete** damage topology (which joint
  is locked, at what angle, and the arm's joint attributes). It is a pure
  function of the diagnostic description and requires **no data**.
- `z_residual ∈ R^8` encodes the **continuous** residual physics. It cannot be
  read off any diagnostic and must be **inferred** from the K calibration
  trajectories.

This factorization separates what is known for free from what must be learned,
and — critically — isolates the only part that changes at deployment.

### A. Topology Encoder

Each joint is described by stable, hand-crafted features rather than a
per-joint lookup embedding, so the encoder generalizes to unseen joints/locks:

    per-joint = [ presence, lock_angle, axis(3), normalized_limits(2), depth ]

where `presence ∈ {0,1}` marks a locked joint, `lock_angle` is its absolute
angle (zero for free joints), `axis` is the joint's unit rotation axis,
`normalized_limits` are its joint limits normalized to [-1,1], and `depth` is
its position along the kinematic chain. A shared MLP encodes each joint
independently, the per-joint codes are mean-pooled along the chain, and a final
linear head produces `e_topology`. Because `e_topology` depends only on the
damage description (never on data), it is available **zero-shot** for any
unseen topology.

### B. Residual Context Inference

`z_residual` is a low-dimensional latent capturing the continuous residual
physics. We use **latent optimization** (variant A):

1. initialize `z_residual = 0`;
2. **freeze** the world model (and policy);
3. minimize the multi-step prediction loss over the K calibration
   trajectories;
4. update only `z_residual` (8 parameters).

Only the residual latent changes at deployment; the world model weights and the
topology encoding are fixed. This makes the causal attribution clean and keeps
deployments comparable. We contrast this with an **amortized encoder**
(variant B), which predicts `z_residual` in a single forward pass from recent
transition history — faster at deployment but, as we show, unable to adapt to
residual parameter levels never seen in training.

### C. World Model

The world model is a deterministic-recurrent network with Gaussian output heads
(GRU + state/reward/continue heads). It predicts

    p(o_{t+1}, r_t, cont_t | o_t, a_t, e_topology, z_residual).

The factorized context `[e_topology, z_residual]` conditions both the recurrent
transition and the prediction heads. The state head predicts a delta from the
current state, so autonomous rollouts remain locally anchored.

### D. Training and Deployment

At training time, the world model and topology encoder are trained jointly on
simulated trajectories over a range of topologies and residual profiles. At
deployment, both are frozen; only `z_residual` is inferred from the K
calibration trajectories via latent optimization.

<!-- TODO: 补 baselines 定义（topology-only / history encoder / monolithic / parameter-matched / residual-only）、实验协议、主结果表。 -->

---

## Baselines (to be finalized)

- **Topology-only**: same WM, `z_residual = 0` (zero-shot, no calibration).
- **History encoder (amortized)**: a GRU predicts `z_residual` from recent
  transitions; same topology encoder and context structure.
- **Monolithic (single continuous descriptor)**: damage encoded as one fused
  descriptor `e_topology + W z`, testing whether factorization itself matters.
- **Parameter-matched**: identical 72-dim context but the residual channel is
  frozen to zero during training, isolating the value of learning structured
  residual use.
- **Residual-only**: only `z_residual` (no topology), testing whether topology
  information matters.
