# 2025--2026 closest top-conference methods: verified reproduction audit

Date: 2026-08-29

## Scope and integrity rule

This audit is limited to verified 2025--2026 top-conference papers with a
direct connection to at least two of: robot world models, non-prehensile
manipulation, dynamics adaptation, low-data system identification, or
fault-tolerant control.  Abstract-level headline numbers are not treated as
mechanism evidence.  The ranking below is based on the paper's controlled
ablations, public code, and compatibility with the calibrated 5-DoF GenkiArm
Push setting.

## Ranked shortlist

### 1. DyWA -- Dynamics-adaptive World Action Model (ICCV 2025)

- Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_DyWA_Dynamics-adaptive_World_Action_Model_for_Generalizable_Non-prehensile_Manipulation_ICCV_2025_paper.pdf
- Project: https://pku-epic.github.io/DyWA/
- Code: https://github.com/jiangranlv/DyWA
- Exact task relation: non-prehensile manipulation under varying object mass,
  table friction, partial observation, and unseen geometry.
- Method: jointly predict the next state and action; encode a history of
  observations/actions into a dynamics embedding; inject that embedding by
  FiLM; train with a privileged RL teacher in simulation.
- Controlled ablation on the hardest unknown-state/single-view track:

| Variant | Seen success | Unseen success |
|---|---:|---:|
| DAgger | 59.9 | 57.5 |
| world model only | 61.6 | 59.4 |
| dynamics adaptation only (RMA) | 65.6 | 57.9 |
| adaptation + FiLM, no world model | 70.0 | 63.7 |
| world model + adaptation, no FiLM | 73.3 | 59.4 |
| full DyWA | 82.2 | 75.0 |

The important lesson is not the 82.2% headline.  World-model supervision alone
adds only 1.7/1.9 points.  The large gain appears only when history-based
dynamics identification, a structured next-state target, and feature-level
conditioning are trained together.  The real-robot result is 68% versus 36%
for CORN on ten unseen objects, but this comparison also includes a different
policy architecture and should not be attributed to FiLM alone.

Reproduction risk: the full repository uses Isaac Gym, PyTorch3D, privileged
teacher training, large object assets, and an older CUDA environment.  A full
Windows reproduction is not the fastest path.  The appropriate first test is
to reproduce only its causal ablation inside the existing MuJoCo pipeline:
history encoder + FiLM-conditioned dynamics, with the current carrier and
evaluation held fixed.

### 2. Simulation Distillation (SimDist, RSS 2026)

- Paper/project: https://sim-dist.github.io/
- Code: https://github.com/CLeARoboticsLab/simdist
- Exact problem relation: rapid adaptation of a pretrained world model to
  changed real dynamics with little data, including contact-rich peg insertion
  and table-leg threading.
- Method: pretrain encoder, latent dynamics, reward, value and policy on large,
  mixed-quality simulation rollouts; at deployment freeze encoder/reward/value
  and update only the dynamics with a short-horizon supervised loss; plan with
  MPPI throughout adaptation.
- Evidence: RSS 2026; real manipulation success is reported over 20 trials per
  condition and adaptation uses roughly 15--30 minutes of real interaction.

The transferable lesson is the separation between task structure and dynamics:
simulation learns representation/reward/value, while deployment adapts only
the component that is actually wrong.  This is stronger than our current
pipeline, which trains a residual but has not demonstrated that its frozen
components supply a useful action-ranking signal.  SimDist also evaluates
improvement through planning success, not RMSE alone.

Reproduction risk: the public stack is IsaacLab-based and substantially larger
than the current project.  The first relevant reproduction is not the entire
robot system.  It is a controlled MuJoCo test that freezes representation and
task heads, fine-tunes only dynamics on fault data, and measures both held-out
prediction and action-ranking/closed-loop change.

### 3. PIN-WM -- Physics-INformed World Models (RSS 2025)

- Proceedings: https://www.roboticsproceedings.org/rss21/p153.html
- Paper: https://www.roboticsproceedings.org/rss21/p153.pdf
- Code: https://github.com/XuAdventurer/PIN-WM
- Exact task relation: few-shot identification of rigid-body dynamics for
  pushing/poking and real-world non-prehensile manipulation.
- Method: identify explicit physical parameters through differentiable physics
  using visual observation loss; create physics-aware digital cousins by
  perturbing identified physical/rendering parameters; train robust downstream
  policies over those cousins.

The useful lesson is that contact uncertainty is represented by identifiable
physical variables and structured randomization, rather than an unconstrained
state residual.  This directly addresses the present GenkiArm symptom: the
learned residual is near zero and unstable because contact timing/friction are
not identified cleanly.

Reproduction risk: the repository is public but its README still marks parts
of data preparation, 2D Gaussian Splatting training, and parameter
identification as pending release.  Full reproduction is therefore lower
priority.  A feasible local analogue is explicit identification of friction,
damping, motor strength and contact parameters in MuJoCo, followed by
physics-aware cousins, while retaining the analytic joint-lock intervention.

## Secondary references, not first reproduction targets

### RINA (ICRA 2025)

- Project/paper: https://hcrlab.gitlab.io/project/rina/
- Relevance: adaptive residual dynamics; a fixed learned neural-operator basis
  is mixed by rapidly adapted linear coefficients to compensate unseen payload
  torques.
- Limitation for this project: quadruped locomotion and payload variation, not
  contact-rich arm manipulation or joint locking; no public code is linked on
  the project page.

### Cross-platform fault-tolerant surfacing (ICRA 2025)

- Project: https://hmmt.ee/research/icra2025/
- Relevance: explicit actuator failures and real-world fault-tolerant control;
  85.7% real success versus 57.1% PID in the reported pool experiment.
- Limitation: underwater surfacing with PPO/LSTM, not a world model or robot-arm
  contact task.  It is useful for evaluation design, not architecture transfer.

## Recommended reproduction order

1. **DyWA-minimal diagnostic**: add a history encoder and FiLM to the existing
   transition model; freeze data, seeds, carrier, capacity-matched baselines and
   evaluation.  Reproduce the six-row ablation above on GenkiArm Push.
2. **SimDist-minimal diagnostic**: reuse diverse intact/fault simulation data to
   pretrain a planning-capable latent model; freeze representation and task
   heads; adapt dynamics only; measure action ranking and closed-loop success,
   not only rollout RMSE.
3. **PIN-WM-inspired diagnostic**: fit a small explicit set of contact and
   actuator parameters and construct physics-aware cousins; compare against a
   parameter-matched neural residual.

The first Go/No-Go should be small and preregistered: seeds 201/211/221, two
training physics conditions, one held-out condition, identical CEM budget, and
no changes after observing the first seed.  A candidate proceeds only if it
improves held-out object prediction by at least 10%, improves top-1 action
regret, and improves terminal Push error in at least 2/3 seeds without free
state regression.  These development seeds must never be reused as final paper
confirmation seeds.

## Immediate conclusion for SI-IPWM

The verified literature does not support adding another unconstrained residual
or renaming the current mask.  The most relevant successful papers obtain large
gains by making the adaptation variable identifiable and useful to decisions:

- DyWA: history identifies dynamics and FiLM changes the policy/world model;
- SimDist: only dynamics is adapted, but a transferred reward/value model turns
  better prediction into better planning;
- PIN-WM: explicit physical identification and structured cousins address
  contact uncertainty.

The next experiment must test one of these mechanisms against the current
SI-IPWM carrier under a frozen, capacity-matched protocol.  Existing No-Go
results remain in the ledger and are not overwritten.
