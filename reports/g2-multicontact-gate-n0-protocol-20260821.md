# Gate N0 Protocol: Set-Valued Multi-Contact Oracle

**Status:** frozen before execution

## Hypothesis

The single-normal Gate M representation fails because the tool capsule and
pusher capsule can contact different block faces or corners. A two-candidate
set of unilateral friction-cone impulses may represent the measured aggregate
pusher impulse without allowing damage information to be ignored.

## Oracle Analysis

1. Snapshot every MuJoCo tool/pusher-to-block contact immediately after
   `mj_step`, including contact position, normal and impulse.
2. Verify extraction orientation: at most 1% of nontrivial exact records may
   have negative impulse along their own normal.
3. Construct deployable tool and pusher capsule candidate frames from the
   fixed-transform chain.
4. With oracle candidate activity, project the measured aggregate impulse onto
   the sum of the two unilateral friction cones.

## Gate

- exact-record negative-normal fraction <= 1%;
- candidate projection RMSE / measured impulse RMS <= 20%;
- failure stops before training;
- pass authorizes only a seed-7 learned set operator, not a contact detector,
  five-seed expansion, control claim or G3.
