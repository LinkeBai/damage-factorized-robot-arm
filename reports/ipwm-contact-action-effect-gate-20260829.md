# IPWM Contact Action-Effect Gate (2026-08-29)

## Question

Can the failed cross-arm object/contact route be replaced by a more
task-directed, arm-invariant mechanism: analytically project a candidate action
through the diagnosed lock, express its end-effector response at contact, and
learn a low-rank operator from that response to the object's next motion?

This is not a tuned continuation of the failed graph head. The hypothesis,
inputs, parameter-matched baseline, seeds, metrics, and decision threshold were
frozen in `config/experiment/ipwm_contact_action_effect_gate_v1.yaml` before the
full development run.

## Protocol integrity

- Calibrated 5-DoF GenkiArm and official 7-DoF Panda MuJoCo models.
- 80 current-contact prefixes per robot, six candidate actions per prefix, and
  three lock branches: 2,880 exact-state paired rows.
- The middle lock is held out on both robots (GenkiArm j3, Panda joint4).
- Prefixes, rather than rows or actions, are grouped across train/validation/test.
- Inputs are current pose/twist and relative geometry, lock depth/angle, and
  current-model Jacobian action responses. Robot identity, solver force, future
  contact, and future state are forbidden.
- Structured model: context-conditioned low-rank 9x3 contact transfer operator.
- Baseline: flat MLP on identical observables.
- Parameter counts are 15,789 versus 15,705 (0.53% difference).
- Primary metrics are held-out-lock object-response RMSE, within-prefix action
  ranking Spearman correlation, and normalized top-1 regret.

Dataset checks passed: maximum intact/locked prefix difference is zero; all
prefixes are in contact; post-step contact is 98.33% for GenkiArm and 100% for
Panda.

## Frozen results

| Seed | Pooled RMSE improvement | Both robots improve | Spearman change | Lower top-1 regret | Gate |
|---:|---:|:---:|---:|:---:|:---:|
| 7 | -25.09% | No | -0.0119 | No | Fail |
| 17 | +4.75% | No | -0.2631 | No | Fail |
| 27 | +6.32% | No | -0.2786 | No | Fail |

The Gate requires at least two positive seeds, at least 10% pooled RMSE
improvement, improvement on both robots, at least +0.10 absolute Spearman gain,
and lower regret. It passes **0/3 seeds**.

## Decision

**No-Go.** The analytic action-response representation is deployable and the
data contract is valid, but the proposed low-rank contact operator does not
turn it into reliable held-out-lock object prediction or action ranking. The
small pooled RMSE gains in two seeds are below threshold, hide a regression on
one robot, and coincide with substantially worse ranking and regret.

No operator rank, hidden size, loss, feature, action perturbation, threshold,
or seed will be tuned. This route cannot enter the paper as a positive core
mechanism and does not authorize the dual-arm Push/Grasp five-seed matrix.

## Scientific implication

For these one-step contact forks, a current-state kinematic Jacobian is not a
sufficient arm-invariant mediator of fault effects. Contact mode, compliance,
and actuation dynamics dominate the action ordering, and they are not captured
by a low-rank instantaneous transfer map. This narrows the remaining search:
another static object head or graph mask is not justified. A future hypothesis
would need an independently observable temporal contact-response variable or a
mechanistic hybrid contact mode, and must again be frozen before evaluation.
