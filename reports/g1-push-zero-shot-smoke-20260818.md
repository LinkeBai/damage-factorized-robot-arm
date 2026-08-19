# G1 goal-directed Push zero-shot smoke (2026-08-18)

## Why this rerun was required

The earlier 15.8% result used random joint excitation inside a scene containing
a block. It did not save contact coverage and used Reach split metadata. This
smoke replaces that setup with a Push-specific immutable split, Push-specific
targets, a shared D2/D3 contact region, and a staged controller that first
approaches the block and then pushes toward the goal.

## Protocol

- K=0 only; no deployment adaptation
- DFWM versus topology-only
- seeds 7, 17, 27 planned; 20 training epochs
- D2 and D3 with `mixed_composition`
- block initial position `(0.24, 0.10)`
- exact `tool_geom`--`block_geom` contact counting
- 200-step goal-directed trajectories

Seed 27 was stopped because seeds 7 and 17 both failed, making the predeclared
2/3 direction gate impossible to pass.

## Results

| Seed | DFWM multi-step RMSE | topology-only | DFWM direction |
|---|---:|---:|---|
| 7 | 0.5746 | 0.4966 | worse |
| 17 | 0.5057 | 0.4273 | worse |

Mean absolute difference (`topology-only - DFWM`) is about -0.0782. The
direction is consistent across both completed seeds.

Coverage was nonzero in both test domains. Across the three evaluation
trajectories per seed, D2 recorded 101 tool-block contact steps and mean block
displacement 0.0607 m; D3 recorded 7 contact steps and mean displacement
0.0133 m. D3 remains a weaker-contact regime and should be improved before a
paper-level benchmark.

## Gate decision

**Zero-shot structured dynamics No-Go under the corrected goal-directed Push
protocol.** The earlier 15.8% random-excitation result does not transfer to the
actual Push workflow. Do not expand it to 5--10 seeds or use it as the paper's
main result.

The next method iteration must explicitly train residual/topology
representations for contact-rich goal-directed trajectories. A supervised
residual-identification or contrastive objective is now a method redesign, not
a minor ablation.
