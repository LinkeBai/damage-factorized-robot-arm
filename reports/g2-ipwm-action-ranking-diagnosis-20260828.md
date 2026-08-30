# SI-IPWM Action-Ranking Diagnosis (Development Seed 27)

## Decision

**NO-GO for direct CEM use and direct carrier-candidate reranking.**

The open-loop SI-IPWM improvement does not currently transfer to MPC because
the learned models do not rank counterfactual actions correctly near the first
contact event. This is now supported by realized MuJoCo branch rollouts rather
than inferred from final episode outcomes.

## Protocol

- Development checkpoint: seed 27 only.
- Domain: `D3__high_damping`.
- Targets: `push_eval_00`, `push_eval_01`, `push_eval_02`.
- Shared candidate sequences for carrier and selective IPWM.
- Decision state: ten steps before the first nominal tool/block or
  pusher/block contact.
- Predicted metric: terminal block-to-goal distance after H10.
- Realized metric: terminal block-to-goal distance after replaying the same
  candidate in MuJoCo from a deterministic matched state.
- Candidate count: 16.

## Root-cause evidence

With H3 or H10 at ordinary replanning states, realized candidate cost variance
is exactly zero while the models predict 2.6--7.3 mm standard deviation. The
controller therefore optimizes model-created differences before or between
contact events.

At the first contact boundary, the true candidates become distinguishable but
their learned ranking is wrong:

| Noise | Target | Carrier Spearman | SI-IPWM Spearman | Carrier regret | SI-IPWM regret |
|---:|:---|---:|---:|---:|---:|
| 0.35 | 00 | -0.623 | -0.341 | 5.93 mm | 5.00 mm |
| 0.35 | 01 | -0.304 | -0.109 | 4.44 mm | 4.44 mm |
| 0.35 | 02 | -0.230 | +0.062 | 5.02 mm | 5.02 mm |
| 0.10 | 00 | -0.764 | -0.179 | 2.76 mm | 2.01 mm |
| 0.10 | 01 | -0.503 | -0.421 | 1.89 mm | 1.61 mm |
| 0.10 | 02 | -0.723 | -0.172 | 2.58 mm | 1.79 mm |

At noise 0.35, neither model contains the realized oracle in its predicted top
quartile for any target. At noise 0.10, SI-IPWM contains it for only one of
three targets. Tightening the action distribution reduces regret magnitude but
does not repair ranking direction.

## Interpretation

The current training/evaluation data establish prediction accuracy on behavior
policy trajectories. MPC asks a different question: rank nearby
counterfactual action sequences, particularly across a discontinuous contact
boundary. Average rollout RMSE is not sufficient evidence for that property.
The negative rank correlations explain why lower object RMSE can coexist with
worse closed-loop terminal distance.

This diagnosis does not invalidate state isolation. It invalidates the implied
bridge from state-isolated prediction to useful planning under the present
training distribution and controller.

## Required method repair

The next development method must add contact-local counterfactual supervision:

1. branch multiple bounded action perturbations from matched pre-contact states;
2. retain paired realized terminal costs and contact/no-contact outcomes;
3. train or calibrate an action-ranking head with a pairwise ordering loss;
4. use the ranking head only inside the carrier-screened candidate set;
5. require positive Spearman correlation on all development targets and a
   paired realized-cost improvement before any untouched-seed expansion.

Reversing a negatively correlated score or selecting a favorable target after
inspection is prohibited as post-hoc repair.

## Local ranker development result

A ridge action-cost calibrator was fitted on four calibration targets and its
regularization was selected on two validation targets. Features are centered
within each candidate set, so the model cannot memorize absolute target
distance. The frozen ranker obtains:

- validation mean Spearman: **0.635**;
- evaluation Spearman: **0.630 / 0.609 / 0.678**;
- evaluation mean regret: **0.582 mm**.

This passes the local branch-ranking diagnostic but not the closed-loop gate.

## Closed-loop transfer of the local ranker

Two deployment interpretations were tested on the same three development
targets:

| Method | Target 00 | Target 01 | Target 02 | Mean |
|:---|---:|---:|---:|---:|
| Carrier MPC | 7.46 mm | 6.81 mm | 18.57 mm | 10.95 mm |
| Replan calibrated ranker every step | 16.48 mm | 23.15 mm | 22.50 mm | 20.71 mm |
| Execute one frozen H10 sequence once | 7.74 mm | 16.94 mm | 23.48 mm | 16.05 mm |

Both calibrated controllers are **NO-GO** relative to carrier MPC. Replanning
violates the sequence-level calibration contract, while one-shot execution
still relies on a distance threshold that does not identify the calibrated
contact phase reliably. No confirmation-seed run is justified.

The next method iteration requires multiple decision-state branches spanning
pre-contact, impact, sustained contact, and separation, plus an observable
contact-phase gate. A single pre-contact state per target is insufficient.

## Contact-phase and strong-baseline continuation

A seven-phase scan showed that counterfactual actions are distinguishable only
near first impact. Before contact and during later free object motion, realized
candidate-cost variance is zero even though both learned models predict
non-zero differences. An observable contact-event trigger therefore replaced
the earlier distance threshold.

The contact-state ranker achieved mean Spearman 0.743 on validation and 0.700
on evaluation branch sets. In a nominal-policy closed loop, the event-triggered
sequence improved all three targets relative to nominal and reduced mean final
distance from 16.99 mm to 14.52 mm. It nevertheless remained worse than the
strong carrier MPC at 10.95 mm and therefore failed the publication gate.

To preserve the strongest baseline, new branches were collected from states
actually induced by carrier MPC, with candidate 0 fixed to the carrier's own
optimized H10 sequence. On three development evaluation targets, direct
SI-IPWM selection found a sequence with lower realized H10 cost than candidate
0 in all three cases. A fitted residual ranker also reduced candidate-0 cost in
all three branch sets. However, neither transferred consistently to full
closed loop:

| Closed-loop method | Target 00 | Target 01 | Target 02 | Mean |
|:---|---:|---:|---:|---:|
| Carrier MPC | 7.46 mm | 6.81 mm | 18.57 mm | 10.95 mm |
| Carrier + fitted residual sequence | 6.18 mm | 12.62 mm | 17.54 mm | 12.11 mm |
| Carrier + direct SI sequence | 5.52 mm | 12.39 mm | 17.54 mm | 11.82 mm |

Both hybrids improve targets 00 and 02 but fail badly on target 01. Their mean
performance remains worse than carrier, so both are **NO-GO**. The result also
shows why favorable short branch ranking cannot substitute for full-episode
evaluation.

Further controller-parameter search on these evaluation targets is prohibited.
The scientifically justified next step is a newly trained contact-action model
using carrier-policy counterfactual branches and a direct relative-cost/ranking
objective, with calibration/validation targets used for development and future
training seeds reserved for confirmation.

## Neural contact-action ranker

A preregistered MLP ranker was trained on 64 carrier-centered candidates per
calibration target. Its inputs include the 14-D state, full carrier H10
sequence, full 50-D candidate residual sequence, carrier/SI predicted costs,
and observable contact-state scalars. Labels are realized MuJoCo H10 branch
costs, not model predictions.

The frozen epoch-80 checkpoint achieved:

| Split | Mean Spearman | Better than carrier candidate 0 | Mean branch improvement |
|:---|---:|---:|---:|
| Calibration | 0.980 | 4/4 | 26.03% |
| Validation | 0.930 | 2/2 | 16.13% |
| Evaluation | 0.871 | 3/3 | 9.58% |

This is strong evidence that local counterfactual action ordering is learnable.
It still does not establish task-level benefit. In the matched C64 closed loop:

| Method | Target 00 | Target 01 | Target 02 | Mean |
|:---|---:|---:|---:|---:|
| Carrier MPC | 11.84 mm | 9.68 mm | 17.52 mm | 13.01 mm |
| Carrier + neural H10 residual | 11.58 mm | 12.75 mm | 16.88 mm | 13.74 mm |

The hybrid improves two targets but regresses target 01 by 3.07 mm. Mean
performance is 5.56% worse than carrier and the catastrophic-regression gate
is triggered. **Closed-loop decision: NO-GO.**

The failure identifies a label-horizon mismatch: H10 realized branch cost can
be optimized accurately, but it does not capture interaction with the carrier
controller after the residual sequence ends. The next protocol must label each
candidate by the full 90-step terminal outcome after resuming the frozen
carrier controller. Short branch cost cannot be reused as a proxy.

## Full-terminal label audit

The first full-terminal v2 dataset generated candidate 0 by running receding
carrier MPC through the next H10 steps on realized MuJoCo states. This made
candidate 0 exactly reproduce the strong baseline and produced encouraging
ranking results, but the full future action sequence is unavailable online at
the contact decision. It is privileged simulator information.

The frozen ranker reached 13.80% mean improvement on the three evaluation
candidate sets, with 2/3 positive targets and only a 0.018 mm regression on the
third. These numbers are retained as a diagnostic upper bound only. They are
**invalid as deployable method or publication evidence** and do not raise the
paper score.

The corrected v3 protocol uses a one-shot H10 CEM sequence generated from the
current observable state and world model. Candidate perturbations and the
ranker therefore receive only information available at deployment. All labels
remain full 90-step terminal outcomes after resuming carrier MPC.

The v3 model was trained only on deployable inputs and full-terminal labels.
It achieved training Spearman 0.968, but validation Spearman was 0.482, below
the frozen 0.5 gate. Both validation targets improved, yet the mean gain was
4.825%, also below the eventual 5% task gate. **v3 decision: NO-GO before
evaluation.** No evaluation-target run or confirmation-seed expansion is
authorized. The principal data limitation is state diversity: four collinear
calibration targets provide many candidate sequences but only four contact
states. The next protocol expands target geometry while retaining the same
ranker architecture.

## Artifacts

- `scripts/diagnose_ipwm_action_ranking.py`
- `runs/g2_ipwm_action_ranking_20260828/dev_seed27_three_targets_contact_h10_c16.json`
- `runs/g2_ipwm_action_ranking_20260828/dev_seed27_three_targets_contact_h10_c16_noise010.json`
- `runs/g2_ipwm_action_ranking_20260828/local_ranker_v1_with_test.json`
- `runs/g2_ipwm_calibrated_control_20260828/dev_seed27_high_three_targets.json`
- `runs/g2_ipwm_calibrated_control_20260828/dev_seed27_one_shot_sequence.json`
- `runs/g2_ipwm_action_ranking_20260828/contact_ranker_v1_with_test.json`
- `scripts/diagnose_ipwm_carrier_policy_ranking.py`
- `runs/g2_ipwm_carrier_policy_ranking_20260828/carrier_residual_ranker_v1_with_test.json`
- `runs/g2_ipwm_calibrated_control_20260828/dev_seed27_contact_event_sequence.json`
- `runs/g2_ipwm_calibrated_control_20260828/dev_seed27_carrier_residual_sequence.json`
- `runs/g2_ipwm_calibrated_control_20260828/dev_seed27_carrier_selective_sequence.json`
- `src/robotarm/models/contact_action_ranker.py`
- `scripts/train_contact_action_ranker.py`
- `runs/g2_ipwm_contact_action_ranker_20260828/ranker_v1_with_test_summary.json`
- `runs/g2_ipwm_contact_action_ranker_20260828/closed_loop_dev_seed27_c64.json`
- `scripts/collect_terminal_action_ranker_branches.py`
- `runs/g2_ipwm_terminal_action_ranker_20260828/ranker_v2_with_test_summary.json`
- `runs/g2_ipwm_deployable_terminal_ranker_20260828/ranker_v3_summary.json`
- `config/experiment/icra_2027_evidence_contract_v1.yaml`
