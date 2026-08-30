# Reproduction-first audit for the 16-day ICRA recovery

Date: 2026-08-30

## Purpose

Determine why closely related papers report large improvements before changing
SI-IPWM.  Each candidate is checked for venue, task, simulator, data, released
code/checkpoints, baseline strength, reported improvement, and local execution.

## Candidate audit

| Work | Venue/year | Task and data | Claimed mechanism | Artifact status | Decision |
|---|---|---|---|---|---|
| DreamFLEX | ICRA 2025 | 4,096 parallel quadrupeds, rough-terrain curriculum, proprioception, simulated joint torque loss/locking, Go1 real tests | explicit fault-vector estimator, beta-VAE latent, fault-conditioned modulation, PPO policy | project page and videos exist; code link is still marked TBU | learn protocol/ablations; cannot faithfully reproduce now |
| Exploring NPM under Multi-Joint Failure | IROS 2024 | PyBullet plus three real tabletop scenarios with locked multiple joints | failure-constrained workspace, kinodynamic action map, sim-in-loop greedy/lazy/random selection | paper available; no source repository identified | reproduce its evaluation logic in our own audited environment after upstream checks |
| Adaptive Compensation for Robotic Joint Failures | 2024 preprint | Isaac Lab Franka drawer opening, permanent/intermittent joint failures, PPO vs numerical IK | POMDP policy trained across failure distributions | linked repository contains only `franka.py`, images and videos; no Isaac Lab, PPO, configs, data or checkpoints | reported 93.6% is not independently reproducible from the release; do not use as a quantitative baseline |
| TD-MPC2 | ICLR 2024 | 104 continuous-control tasks; online state/pixel and offline multi-task data | latent dynamics + reward + distributional Q + policy prior, policy-guided CEM | complete code, 300+ checkpoints and datasets | upstream minimal reproduction completed; next transfer baseline |
| DINO-WM | 2024 arXiv, cs.RO | offline PointMaze/Wall/Reach/PushT/Rope/Granular; PushT uses 18,500 trajectories/samples as described by the paper | frozen DINOv2 patch features, action-conditioned transformer dynamics, latent-goal CEM | complete code; OSF checkpoint 953 MB, PushT data 2.79 GB | checkpoint download started; run upstream PushT before any visual transfer |
| DyWA | ICCV 2025 | IsaacGym tabletop rearrangement; 323 DexGraspNet training objects and 50 unseen evaluation objects | history-conditioned dynamics adaptation + future-state prediction + PPO teacher/DAgger student + variable impedance | complete public repository and pretrained-policy links; Linux/IsaacGym/CUDA 11.3 stack | code-level audit accepted; upstream run pending because the current Windows stack is incompatible |
| fault-locomotion-isaaclab / MoE Fault Locomotion | arXiv 2026 | Isaac Lab, 4,096 parallel Go2/Aliengo/Pegasus environments, 29 healthy/failure modes | explicit joint status + five-step history + concurrent estimator/RMA + PPO/MoE + fault-specific rewards | complete BSD repository; commit `a85654c`; no Isaac Lab installation on this host | source compiles; performance reproduction is blocked by a missing simulator, so no paper result is counted |
| LeWorldModel / stable-worldmodel | 2026 arXiv | pixel PushT, TwoRoom, Reacher and OGBench Cube; released datasets and checkpoints | end-to-end JEPA with next-embedding prediction and Gaussian latent regularization, latent CEM planning | complete code/checkpoints; PushT environment and local data collection run on this host | environment-level reproduction completed; checkpoint evaluation is the next gate |

## Completed upstream reproduction: TD-MPC2

- Repository: `nicklashansen/tdmpc2`, commit
  `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`.
- Environment: Windows, Python 3.12, PyTorch 2.11 CUDA, MuJoCo 3.12,
  DM-Control, `MUJOCO_GL=glfw`.
- Compatibility corrections: Gymnasium pinned to 0.29.1; official default
  `MUJOCO_GL=egl` replaced by `glfw` for Windows DM-Control.
- Task: official `cartpole-balance`, state observations, 1.20M-parameter model.
- Budget: 5,000 online steps, seed 1, CEM 128 samples/16 elites, 8 policy
  trajectories, compilation and W&B disabled only for local reproducibility.
- Initial evaluation return: 239.9.
- Evaluation return at 2,500 steps: 203.6.
- Evaluation return at 5,000 steps: 657.6.
- Final training return: 666.1.
- Relative final-vs-initial evaluation increase: 174.1%.

This is a short-run reproduction, not the paper's 104-task claim.  It proves
that the upstream algorithm and its large learning effect run on this machine.

## Why TD-MPC2 improves much more than current SI-IPWM control

TD-MPC2 does not optimize state RMSE and bolt CEM on afterwards.  Its latent
dynamics, reward model, distributional value functions and policy prior are
trained together.  Planning samples both Gaussian candidates and policy-prior
trajectories, then scores them with predicted reward plus terminal value.  The
current SI-IPWM pipeline is strongest on open-loop state/object prediction and
uses an external task cost; therefore lower RMSE need not preserve action rank
or long-horizon value.  This is now a concrete implementation hypothesis to
test, not a proposed paper claim.

## Second-wave code audit: where the reported large gains actually come from

### DyWA

The reported gain cannot be attributed to a world-model head alone. The public
protocol trains a privileged teacher with PPO for 200k iterations and then a
student with DAgger for 500k iterations. It randomizes object mass, scale,
friction, restitution, torque, point clouds, and goals; uses 323 training objects;
and acts through end-effector residuals plus learned impedance. Its loss combines
imitation, world-model, and dynamics-adaptation objectives. A fair transfer must
therefore separate four factors: teacher/DAgger scale, history adaptation,
future-state auxiliary loss, and impedance action space. Copying only the
world-model auxiliary loss would not reproduce the claimed gain.

### fault-locomotion-isaaclab

The checked-out upstream commit is `a85654c2955392242f0ee0333428104f72256bd0`.
Its default Go2 configuration uses 4,096 parallel environments, five-frame
observation history, explicit 12-dimensional joint status, a concurrent TCN state
estimator, asymmetric PPO privileged state, action/observation noise, and dense
tracking, stability, contact, smoothness and failure-specific rewards. It
implements 29 selectable healthy/single-/multi-joint motor-disable patterns by
zeroing actuator stiffness and damping, including failures switched during an
episode. This is closer to power loss than to our hard lock
`q_j = qbar_j, qdot_j = 0`; it is a useful training/evaluation recipe, but is not
a drop-in baseline for the paper's structural-lock claim.

Verification on this host:

- repository Python sources pass `compileall`;
- Isaac Lab is not installed (`ISAACLAB_CLI_NOT_FOUND`);
- consequently no success/reward number from this project is treated as reproduced.

### LeWorldModel / stable-worldmodel

The upstream LeWM repository was checked out at commit `8a2c595`. The initial
`stable-worldmodel[env]` installation failed because Gymnasium's all-environment
extra attempted to compile legacy Box2D on Windows/Python 3.12. A minimal PushT
installation was then built without Box2D while preserving CUDA PyTorch
2.11.0+cu128. The official `swm/PushT-v1` environment reset successfully, and
the built-in weak expert collected two 64x64 episodes into a Lance dataset
(387,680 bytes, four files). This is an environment/data-pipeline reproduction,
not yet a reproduction of LeWM's planning success.

The released 72,290,721-byte PushT checkpoint was then downloaded, instantiated
with the versions pinned by the upstream `uv.lock` (`stable-pretraining==0.1.7`,
`transformers==4.57.6`, `tokenizers==0.22.2`,
`huggingface-hub==0.36.2`), and loaded with strict state-dict checking. All keys
matched for the 18,034,478-parameter model. A one-episode, locally collected
diagnostic with 64 CEM candidates, five CEM updates, horizon 5, goal offset 25
and budget 50 completed successfully (1/1 success; first CEM solve 0.404 s).
This proves the released checkpoint-planner-control path runs locally, but 1/1
is **not** a performance estimate and must not be compared with the paper's
reported aggregate success rate.

The diagnostic was then expanded to ten locally collected episodes and paired
with a seeded random policy. At goal offset 25, LeWM succeeded on 4/10 episodes
and random on 1/10; the three discordant wins all favored LeWM, but an exact
two-sided sign/McNemar test gives `p=0.25`, so this is directional rather than
decisive evidence. At goal offset 50, LeWM succeeded on 1/10 and the final
frozen seeded-random run succeeded on 0/10. This one-episode difference is not
statistically useful. Increasing CEM to the author's 300 samples and 30
iterations did not
recover episode 1 at offset 50. Therefore this local test supports that the
checkpoint executes and can help at short offsets, but it does **not** reproduce
the paper's aggregate advantage and does not justify prioritizing LeWM as the
paper's main transferred baseline.

Frozen machine-readable outputs:

- `results/reproductions/lewm_pusht_lewm_go25_n10.json`
- `results/reproductions/lewm_pusht_random_go25_n10.json`
- `results/reproductions/lewm_pusht_lewm_go50_n10.json`
- `results/reproductions/lewm_pusht_random_go50_n10.json`

Two upstream compatibility findings are preserved rather than hidden:

- the released training dataset is 13,136,247,974 bytes compressed, so it was
  not downloaded for this time-bounded smoke test;
- the current LeWM/stable-worldmodel stack fails for solver batch size greater
  than one because the goal embedding lacks the CEM sample dimension. The
  official solver config uses batch size one, which is the setting used for the
  successful run.

The exact diagnostic is implemented in
`scripts/reproduce_lewm_pusht_smoke.py`.

This project is useful for the diagnosis because it supplies a standardized
contact-rich task, identical checkpoint-evaluation machinery for LeWM and several
baselines, and explicit success criteria. It does not model structural joint
failure, so it is first a control-aligned world-model reference and only later a
transfer candidate.

## Diagnosis matrix required before IPWM innovation

| Check | Interpretation if it fails | Next action |
|---|---|---|
| upstream author's task and checkpoint | environment/dependency/reproduction problem | fix or reject artifact; do not change IPWM |
| healthy original 5-DoF task | adapter, reward or controller problem | repair unified task protocol |
| known hard-lock fault | failure representation or training-distribution problem | compare diagnosed fault status and history inference |
| held-out joint/lock angle | structural generalization problem | test hard projection and factorized propagation |
| action ranking despite lower RMSE | planning-objective mismatch | add value/reward/ranking supervision, not another free residual |

Only a mechanism that passes the matched ablation on the last two rows is eligible
to become a claimed IPWM innovation.

## Original 5-DoF TD-MPC2 transfer: adapter smoke test

A committed adapter now exposes the original `sim/assets/arm_push.xml` through
the four-return environment API expected by TD-MPC2. It uses a 33-D observation
(14-D arm/block state, end-effector, target and relative geometry, plus known
lock mask/angle), five normalized joint actions, and an object-goal reward with
progress, approach, action-cost and success terms. Targets are sampled at least
25 mm from the initial block and success tolerance is 10 mm, preventing the old
50 mm initial-success leakage. Unit tests verify both the API and exact D3 hard
lock (`q3=-0.5`, `qdot3=0`) under repeated nonzero commands.

The first intact-arm smoke run used TD-MPC2 model size 1, seed 1, 1,200 steps,
64 CEM candidates, eight elites and four policy trajectories. It completed with
1,214,649 trainable parameters and a 1,200-transition CUDA replay buffer. Initial,
600-step and 1,200-step evaluation returns were -22.00, -20.52 and -23.73;
success remained 0%. Because TD-MPC2 reserves the first 1,000 steps for seed
data, this run contains only about 200 learning steps and is an integration
pass, not a method result. The next gate is a 5,000-step intact-arm run before
any hard-lock training.

The matched 5,000-step intact run with uniform random seed collection also
failed: evaluation return changed from -22.00 at step 0 to -17.73 at step
2,550, while success stayed 0%. A separate 100-episode/15,000-step coverage
audit found zero tool-block contact episodes and exactly zero block motion.
Thus this failure cannot identify push dynamics and is not evidence against a
world-model architecture.

Inspection then found a geometry bug in the historical goal controller. Its
fixed `+x` waypoint assumes a single push direction; for the current leftward
targets it approached from the target side of the block. The repaired
direction-aware controller starts behind the block relative to the desired
motion and pushes along the block-to-target direction. In a frozen 20-episode
screen, uniform random seeding again produced 0% contact, 0% success and zero
block displacement, whereas direction-aware seeding produced 100% contact,
75% controller success and 38.8 mm mean maximum block displacement. This is a
data-coverage/control result, not a learned-model score.

An initial TD-MPC2 transfer silently bypassed that policy because its tensor
wrapper overwrote `env.rand_act()` with `action_space.sample()`. After fixing
the wrapper and enabling episodic termination, the guided 5,000-step run
completed. Seed episodes included successful pushes, but frozen policy
evaluation still scored 0% success: return changed from -19.71 at step 0 to
-18.04 at step 2,591. The matched random run was -17.73 and 0% at step 2,550.
Therefore the current gate is **No-Go for a TD-MPC2 learned-control advantage at
5k steps**, while the contact-coverage repair itself is a verified Go. The next
diagnostic is a behavior-cloning sanity check and/or a longer official-budget
run; IPWM must not be credited with, or blamed for, this result.

Frozen machine-readable outputs:

- `results/reproductions/tdmpc2_original_arm_random_seed_coverage_seed1_n20.json`
- `results/reproductions/tdmpc2_original_arm_directional_seed_coverage_seed1_n20.json`
- `results/reproductions/tdmpc2_original_arm_guided_5k_seed1.json`

A behavior-cloning sanity check then trained a two-layer MLP on 40 directional
demonstrations (5,101 transitions). The teacher succeeded on 80% of those
training episodes. On 20 unseen target seeds, BC reached contact in 100% of
episodes but succeeded in only 25%; mean terminal goal error was 72.1 mm. This
shows that the observation/action interface carries enough information to learn
contact approach, while one-shot imitation suffers severe post-contact
distribution shift and overshoot. Together with the TD-MPC2 result, the next
transfer should reproduce teacher-data aggregation or corrective rollouts and
control-aligned value/ranking supervision. Merely increasing transition-model
capacity is not supported by the evidence.

BC output: `results/reproductions/original_arm_bc_sanity_seed1_train40_eval20.json`.

## Minimal hard-lock recovery screen

The smallest non-trivial fault slice was screened before adding another learned
module. A healthy directional controller was executed after a hard lock in two
forms: **fault-unaware**, which retains the intact IK references while the
failed action coordinate is physically zeroed, and **fault-aware constrained
IK**, which recomputes both waypoints under the known lock constraint. Success rates
over three 10-episode screens were:

| Lock | Fault-unaware | Fault-aware constrained IK | Absolute gap |
|---|---:|---:|---:|
| D2 | 10%, 20%, 20% | 90%, 90%, 90% | 70--80 pp |
| D3 | 0%, 0%, 0% | 100%, 100%, 100% | 100 pp |
| D4 | 0%, 0%, 0% | 100%, 100%, 100% | 100 pp |

The failure modes differ materially. D3 fault-unaware control never contacts
the block, whereas D4 contacts in every episode but pushes far past/wrong with
roughly 81--84 mm terminal error. Thus the minimal problem contains both
pre-contact reachability repair and post-contact action-effect repair. The
The constrained-IK baseline reaches roughly 9--10 mm terminal error.

These numbers are **not learned-model results**. They establish recoverable
headroom for a known-fault constrained controller. In addition, the three
screens use partially overlapping target-seed ranges, so they are a development
robustness screen rather than three independent statistical seeds. Formal
claims require disjoint frozen target sets, independent training seeds, and a
learned or calibrated contact-effect method that improves on the shared
constrained-IK carrier without receiving held-out outcomes.

The resulting narrow research question is: can a hard-constrained,
counterfactual **action repair** model use 1--5 fault interactions to transform
an intact push action into a feasible free-joint action that preserves the
desired end-effector/contact-object effect after an unseen single-joint lock?
This removes full-policy relearning and full-scene prediction from the primary
claim. The fair baselines are intact action with failed coordinate zeroed,
constrained Jacobian/IK repair, BC/DAgger adaptation, and an unconstrained
action-residual model.

## Transfer matrix before method innovation

1. **TD-MPC2 healthy original arm:** establish whether the adapter and reward
   produce a strong Push controller at all.
2. **TD-MPC2 fault-conditioned:** append diagnosed lock identity/angle, zero the
   failed action coordinate, and train on the same fault distribution.
3. **TD-MPC2 held-out fault:** D2/D4 training, D3 testing, identical target split.
4. **Current carrier/IPWM under the same reward and evaluation:** same episodes,
   success definition, action budget and wall-clock accounting.
5. **DINO-WM upstream PushT:** reproduce released checkpoint planning before
   considering eye-to-hand patch features for the original arm.

Only after these rows exist do we decide whether SI-IPWM needs a control-aligned
reward/value head, a policy prior, a different representation, or a narrower
claim.  A transferred baseline that already solves held-out locks would show a
method problem; failure of all strong baselines would instead point to task,
reward, reachability, observation or simulator defects.

## Immediate gate

- Do not modify IPWM while TD-MPC2 healthy-task integration is unverified.
- Do not compare percentages across Cartpole, quadruped locomotion, drawer
  opening and our Push task as if they were the same metric.
- Preserve all upstream versions, commands, logs and failures.
- Advance a method only after at least two development seeds improve the same
  control metric under the same task protocol.
