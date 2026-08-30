# IPWM Formal Invariants and Guarded-Fallback Bound

## Scope

These results formalize what the current SI-IPWM implementation guarantees independently of learned-model accuracy. They do not imply lower prediction error or better task performance.

## Proposition 1: analytic feasibility and idempotence

For diagnosis `d=(m,qbar)` with binary mask `m`, define

`Pi_d(q)=(1-m)⊙q+m⊙qbar`, `Pi_d(qdot)=(1-m)⊙qdot`, and `Pi_d(a)=(1-m)⊙a`.

Then `Pi_d(Pi_d(x))=Pi_d(x)`. Every locked coordinate satisfies `q_j=qbar_j`, `qdot_j=0`, and every locked action satisfies `a_j=0`. If each learned transition is followed by `Pi_d`, these equalities hold at every finite rollout depth by induction, regardless of network weights.

Evidence: `TopologySurgery.project_state/project_action`, solver-native lock tests, and the random-batch idempotence/multi-step tests.

## Proposition 2: published-state non-interference

Let the carrier and intervention branches keep private recurrent states. At each step publish carrier robot coordinates and intervention object coordinates, followed by the same analytic projection. For identical initial state, action sequence, diagnosis and carrier parameters, the published robot trajectory equals the standalone carrier robot trajectory at every depth. Any deterministic forward kinematics of the published joint state is also identical.

This is an isolation guarantee relative to the carrier, not to ground truth. The intervention branch may retain arbitrary internal robot--object coupling; its robot output cannot rewrite the published carrier robot block.

Evidence: `SelectiveInterventionRollout` and exact 25-step random-batch tests.

## Proposition 3: conditional guarded-fallback bound

Let `A` denote acceptance of an intervention proposal, `S` the event that its certificate is valid, and `J` the finite-horizon task cost. The guarded policy publishes the carrier action exactly on rejection. Assume accepted, valid proposals obey `J(pi_i)-J(pi_c) <= L epsilon_H`, where `epsilon_H` is a measured horizon-level certificate error; assume any false acceptance has bounded excess cost at most `C`. Then

`E[J(pi_g)-J(pi_c)] <= L E[epsilon_H 1(A and S)] + C P(A and not S)`.

Proof: rejection contributes exactly zero because `pi_g=pi_c`; partition accepted events into valid and false-accepted sets and apply the two assumed bounds.

The inequality is useful only when `epsilon_H`, false-accept probability and `C` are estimated on disjoint validation data. The current repository has exact fallback semantics but does not yet have sufficient calibration evidence to claim a small numerical bound.

Evidence: `guarded_action` and exact-selection tests. This proposition motivates reporting coverage, false accepts and worst accepted degradation rather than describing the guard as a generic safety guarantee.

## Publication boundary

The three propositions support exact feasibility, carrier-relative non-interference and a conditional risk decomposition. They do not repair the observed action-ranking or closed-loop No-Go and must not be presented as universal robustness or performance theorems.
