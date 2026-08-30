# Gate M1b Protocol: Exact Contact Impulse Decomposition

**Status:** frozen before M1b oracle evaluation

## Correction

Gate M v1 read contact forces after damage projection and `mj_forward`, and
treated total object velocity change as pusher impulse. M1b snapshots contact
forces immediately after `mj_step`, separates tool/pusher and table impulses,
and includes the block slide-joint damping used by the MJCF.

For mass `m`, damping `b`, time step `dt`, and total measured planar contact
impulse `J`, the oracle update is:

```text
v_next = (v + J / m) / (1 + dt b / m)
```

## Gate

- implicit momentum reconstruction RMSE must be at most `0.001`;
- exact pusher impulse projected on the deployment contact basis must have at
  most 5% materially negative normal samples;
- failure stops M1b before learned-model training;
- pass authorizes one seed learned pusher-impulse training with the analytic
  damping operator, not geometric mode detection or five-seed expansion.
