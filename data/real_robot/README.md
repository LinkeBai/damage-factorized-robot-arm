# Original 5-DoF real-robot Push packet

This directory contains only schemas and templates until measurements are
collected. Blank cells are not evidence and must never be imputed.

Before the first method trial, copy `session_manifest_template.yaml` to a new
session-specific manifest, fill every field, compute the randomized schedule
hash, and set the freeze record. The binary success threshold is fixed at 30 mm
to match the simulation protocol; changing it requires a dated deviation and
both old- and new-threshold results.

## Minimum field sequence

1. Photograph the arm, gripper, block, table, both fixed eye-to-hand cameras,
   calibration target, emergency stop, and cable routing.
2. Record joint IDs/directions/limits and verify low-speed stop behavior.
3. Synchronize the overhead and horizontal videos with the control clock using
   one visible event at the start and end of each session.
4. Run intact, D2, and D3 low-amplitude identification probes before contact.
5. Verify constrained-IK reach/contact at least 4/5 times per condition before
   attempting a learned-method comparison.
6. For the primary method comparison, interleave `nominal` and
   `global_matched` under the same `pair_id`, physical reset position, target,
   lock, and action library. This follows the simulation result that actually
   passed the stable control-signal gate. Add `si_ipwm` as the third row when
   time permits; it is an attribution control, not the presumed winner. Record
   failures and aborted trials; never delete them.
7. Reserve the final 45 minutes for opening every file, hashing, and copying the
   packet to two independent drives.

## Required trial fields

`push_trials_template.csv` is parsed by `scripts/analyze_real_robot_push.py`.
Every non-aborted row requires measured lock error, reach/contact labels,
terminal error, success, both video paths, and the control-log path. Every
aborted row requires a `failure_code`.

Run the strict validity and file gate from the repository root:

```powershell
python scripts/analyze_real_robot_push.py data/real_robot/push_trials.csv `
  --reference-method nominal --candidate-method global_matched `
  --require-files --output results/real_robot/push-summary.json
```

The analyzer labels fewer than ten complete reference/candidate pairs as `pilot`.
Ten or more pairs only changes the evidence level to `formal`; it does not imply
statistical significance or a positive result.

## Success definition

Freeze the success threshold before method labels are inspected. Always report
continuous terminal error, reach rate, contact rate, maximum lock error, aborts,
and failure codes even when a binary success rate is shown.

The two visual sources are fixed eye-to-hand cameras. Do not describe either as
eye-in-hand.
