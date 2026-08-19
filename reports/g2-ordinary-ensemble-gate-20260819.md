# G2 Ordinary Deep-Ensemble Gate (2026-08-19)

## Question

Does the topology-conditioned ensemble outperform an otherwise identical deep
ensemble that never observes damage identity?

## Frozen Protocol

- Config: `config/experiment/g2_push_ensemble_v1.yaml`
- Task: corrected Push; D2 and D3 mixed-composition evaluation
- Seeds: 7, 17, 27, 37, 47
- Three members per method; 20 epochs; 150 steps
- Same member architecture, parameter count (450,906), optimizer, training
  trajectories, evaluation trajectories, and rollout horizon
- Only ablation: the ordinary ensemble receives the same intact descriptor for
  every trajectory, while the structured ensemble receives the diagnosed
  topology descriptor

## Results

| Seed | Mean structured improvement over ordinary |
|---:|---:|
| 7 | 2.76% |
| 17 | 6.36% |
| 27 | 0.29% |
| 37 | 8.21% |
| 47 | -5.26% |

The five-seed mean is **2.47%**. A paired seed bootstrap with 50,000 resamples
gives a 95% interval of **[-1.83%, 6.38%]**. The interval crosses zero and one
seed reverses direction.

## Decision

The current topology-conditioning mechanism does not pass the G2 method gate.
The earlier 30.7% improvement over a parameter-matched wide single model is
evidence for ensemble averaging versus a single model, not evidence that the
topology structure itself is responsible.

Per the preregistered V6 rule, this triggers the benchmark Pivot. Do not tune the
topology encoder against these test results. A domain-randomized ensemble may be
added to complete the benchmark, but the current method cannot be presented as
a statistically established structured-dynamics contribution.

Source data:

- `results/final/g2_structured_vs_ordinary_5seed.csv`
- `results/final/g2_structured_vs_ordinary_5seed.json`
