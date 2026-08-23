# Related-work audit for the ICRA manuscript

## Closest published work

1. Finn and Levine, Deep Visual Foresight for Planning Robot Motion, ICRA
   2017: learned visual dynamics for pushing and planning; no damaged manifold.
2. Nagabandi et al., Neural Network Dynamics for Model-Based Deep RL, ICRA
   2018: recurrent dynamics and online improvement; no failure projection.
3. Cully et al., Robots that can adapt like animals, Nature 2015: behavior
   recovery after damage rather than contact-world-model adaptation.
4. Yang et al., Fault-Aware Robust Control via Adversarial RL, IEEE CYBER
   2021: policy robustness across joint damage.
5. Kumar et al., RMA, RSS 2021: latent extrinsics from recent history; no
   diagnosed lock projection or object-intervention routing.
6. Allevato et al., TuneNet, CoRL 2020: one-shot residual identification.
7. Bauza et al., Data-Efficient Precise Pushing, CoRL 2018: few-sample pushing.
8. Cong et al., Self-Adapting Recurrent Models for Object Pushing, IROS 2020.
9. Kim et al., SE(2)-Equivariant Pushing Dynamics, CoRL 2022/2023.
10. Richards et al., Adaptive-Control-Oriented Meta-Learning, RSS 2021.

## Defensible novelty gap

No audited paper combines all four elements: a diagnosed unseen joint-lock
intervention, exact analytic state/action projection, a support-aware
robot-to-object intervention residual, and observable few-shot physical context
with exact zero-context fallback. Claim this combination and its matched
evidence, not priority over all fault-tolerant robotics.

## Claim discipline

- Use contact-aware block-coordinate, not strict forward block triangular.
- Use physical-context intervals, not rollout-risk probability.
- K25 is frozen and effective; performance is not claimed monotonic in K.
- Until real-arm rows are filled, state simulation evidence only.
