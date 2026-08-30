# Hybrid Fault-Counterfactual Novelty Audit (2026-08-29)

## Candidate hypothesis

The remaining candidate is a fault-conditioned hybrid counterfactual world
model: analytic lock projection and state isolation define the safe carrier;
the learned component predicts contact-mode survival under each candidate
action and a mode-conditioned cumulative object response. Training and testing
use paired intact/locked forks, and acceptance depends on action ranking and
regret as well as state error.

## Close prior mechanisms

1. Johnson, Burden, and Koditschek, *A Hybrid Systems Model for Simple
   Manipulation and Self-Manipulation Systems* (2015), already formalize
   make/break contact as a hybrid dynamical system with consistency guarantees:
   https://arxiv.org/abs/1502.01538
2. Saxena, LaGrassa, and Kroemer, *Learning Reactive and Predictive
   Differentiable Controllers for Switching Linear Dynamical Models* (2021),
   already learn switching conditions for contact dynamics and use them for
   predictive control: https://arxiv.org/abs/2103.14256
3. Bianchini, Halm, and Posa, *Simultaneous Learning of Contact and Continuous
   Dynamics* (CoRL 2023), jointly identify contact and continuous dynamics from
   contact-rich trajectories:
   https://proceedings.mlr.press/v229/bianchini23a.html
4. Allen et al., *Graph Network Simulators Can Learn Discontinuous, Rigid
   Contact Dynamics* (CoRL 2023), show that general graph simulators can learn
   discontinuities without explicit contact modules:
   https://proceedings.mlr.press/v205/allen23a.html
5. Pizzuto and Mistry, *Physics-penalised Regularisation for Learning Dynamics
   Models with Contact* (L4DC 2021), incorporate contact physics constraints
   into learned dynamics:
   https://proceedings.mlr.press/v144/pizzuto21a.html
6. Omar and Khadiv, *Learning to Act Through Contact* (L4DC 2026), use an
   explicit contact representation across embodiments and tasks:
   https://proceedings.mlr.press/v331/omar26a.html

## Differentiation that is defensible

The novelty cannot be “we predict contact modes” or “we use a hybrid world
model.” Those claims are occupied. The potentially differentiating unit is the
following conjunction, which must be evaluated as one mechanism rather than a
list of modules:

1. the structural intervention is diagnosed but unseen during training;
2. the lock changes the feasible manifold and is enforced analytically;
3. a protected carrier prevents learned fault adaptation from rewriting
   published unaffected robot coordinates;
4. the model predicts how the structural intervention changes contact-mode
   survival under alternative actions;
5. conditional response predictions are accepted only if they improve
   counterfactual action ordering and closed-loop decisions.

No reviewed work was found to establish this exact fault-intervention/safe-
carrier/mode-survival/action-ranking chain. This is a differentiation claim,
not a “first” claim; the search is not a proof of absence.

## Novelty score boundary

- Contact classifier plus two regression heads: **3.0--3.3/5**, component
  combination, regardless of performance.
- Unified fault-conditioned hybrid factorization with paired counterfactual
  training, exact isolation guarantee, and decisive ranking evidence:
  **3.6--3.9/5** if it passes.
- A score above 4.0 for novelty would additionally require a nontrivial result,
  such as an identifiable intervention-effect factorization or a planning
  regret bound whose terms are experimentally measurable and tighter than an
  unstructured model. No such result currently exists in the repository.

## Decision

The candidate is sufficiently differentiated for one small falsification Gate,
but not sufficiently novel to be called a strong contribution before that
Gate and a formal result succeed. The Gate must compare against both a
parameter-matched unstructured multi-task predictor and an equally supervised
non-mixture predictor. If mode supervision alone explains the gain, the result
is an auxiliary-task ablation, not a new world-model mechanism.
