# Calibrated GenkiArm three-seed interim result

Date: 2026-08-30

This is the authoritative interim result for preregistered seeds 107, 117 and
127 on `sim/assets/genkiarm_push.xml`.  It supersedes the two-seed interim
report but does not replace the required five-seed analysis.

| Method | Seed 107 | Seed 117 | Seed 127 | Mean | Positive seeds |
|---|---:|---:|---:|---:|---:|
| routed selective SI-IPWM | +0.6037% | -0.7836% | +0.7182% | +0.1794% | 2/3 |
| raw selective SI-IPWM | +1.4198% | -1.9133% | +2.5459% | +0.6841% | 2/3 |

The hierarchical paired-bootstrap 95% intervals are:

- routed: `[-0.7446%, +0.7972%]`;
- raw selective: `[-1.8698%, +2.5017%]`.

Both methods retain exactly zero free-joint regression and zero
locked-coordinate violation in all three completed seeds.  The state-isolation
invariant therefore replicates, but object-prediction superiority does not.

The preregistered performance gate fails: mean improvement is below +5%, only
2/3 seeds are positive, and both intervals cross zero.  Physical-context
training selected epoch 0 for seed 127 and supplied no additional gain.  The
correct interim decision is **No-Go for performance superiority; replicated
mechanism evidence only**.

Authoritative machine-readable artifact:
`runs/g2_ipwm_genkiarm_confirmation_v2/three_seed_interim_summary.json`.

