# G2 Complete Gate Report

**Date**: 20260819
**Hypothesis H-ZST**: topology descriptor improves prediction on held-out topology (D3)

---

## Experiment 1: Original G2 (D2+D3 in training)

Structured vs ordinary ensemble, both D2 and D3 in training set.

| Seed | D2 improvement |
|---:|---:|
| 7 | +3.56% |
| 17 | +4.99% |
| 27 | +0.69% |
| 37 | +4.55% |
| 47 | -4.90% |

Mean: **+1.78%**  95% CI: **[-1.80%, +4.53%]**  4/5 positive
**Decision: NO-GO** — CI crosses zero

---

## Experiment 2: Held-Out Topology (D3 absent from training)

### Primary: D3 mixed_composition (unseen topology)

| Seed | structured RMSE | ordinary RMSE | improvement |
|---:|---:|---:|---:|
| 7 | 0.5071 | 0.5120 | +0.95% |
| 17 | 0.4826 | 0.4727 | -2.09% |
| 27 | 0.6080 | 0.5771 | -5.36% |
| 37 | 0.5157 | 0.5136 | -0.41% |
| 47 | 0.6669 | 0.7171 | +7.00% |

Mean: **+0.02%**  95% CI: **[-3.38%, +3.70%]**  2/5 positive
**Decision: NO-GO**

### Control: D2 mixed_composition (seen topology)

| Seed | structured RMSE | ordinary RMSE | improvement |
|---:|---:|---:|---:|
| 7 | 0.5116 | 0.5388 | +5.05% |
| 17 | 0.4860 | 0.4735 | -2.63% |
| 27 | 0.6351 | 0.6099 | -4.12% |
| 37 | 0.5221 | 0.5271 | +0.95% |
| 47 | 0.7282 | 0.7766 | +6.23% |

Mean: **+1.10%**  95% CI: **[-2.51%, +4.70%]**  3/5 positive
**Decision (control): NO-GO**

---

## Overall G2 Gate

**NO-GO** — Neither D3 held-out nor D2 control passes the CI gate. Topology conditioning provides no statistically stable benefit. Per V6 plan, proceed to benchmark/negative-result paper.

---

## Failure Analysis

1. **Conditioning redundancy (original G2)**: with D2+D3 both in training, ordinary ensemble learns condition from trajectory data; topology descriptor is redundant.

2. **Weak zero-shot generalization (heldout-topology G2)**: even with correct D3 descriptor at test time, structured ensemble gains only marginal improvement over ordinary ensemble on D3. The descriptor provides correct topology prior but the model trained only on D2+intact cannot leverage it to accurately predict D3 dynamics.

3. **Root cause**: the topology descriptor encodes which joint is locked, but the world model needs to have learned the *dynamics consequences* of locking that joint. Without D3 training data, the model has no dynamics basis to associate with the D3 descriptor.

## Conclusion

The structured topology-conditioned ensemble does not demonstrate a statistically stable advantage over an ordinary deep ensemble under either experimental protocol. Per V6 preregistered Pivot rules, the project transitions to benchmark/negative-result framing for ICRA 2027.