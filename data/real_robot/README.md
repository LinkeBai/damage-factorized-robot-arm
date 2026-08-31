# Original 5-DoF real-robot Push packet

This directory contains only schemas and templates until measurements are
collected. Blank cells are not evidence and must never be imputed.

Before the first method trial, copy `session_manifest_template.yaml` to a new
session-specific manifest, fill every field, compute the randomized schedule
hash, and set the freeze record. The binary success threshold is fixed at 30 mm
to match the simulation protocol; changing it requires a dated deviation and
both old- and new-threshold results.

Run the hard preflight immediately before trial 1. Exit code 0 and the
mode-specific `LEVEL_A_TRIALS_MAY_START` or
`LEVEL_B_METHOD_TRIALS_MAY_START` authorization are both required;
`--skip-path-existence` is only for schema tests and is forbidden during the
real session:

```powershell
python scripts/audit_real_robot_preflight.py data/real_robot/session_20260901.yaml `
  --mode level_a --schedule data/real_robot/level_a_schedule_frozen.csv `
  --output results/real_robot/preflight-audit.json
```

The formal nominal/global comparison also requires a validated action-interface
bridge. The simulator uses generalized motor force while the arm accepts servo
goal positions; these are not interchangeable. See
`reports/real-robot-action-interface-audit-20260831.md`. Without the bridge and
common action-library hash, collect only the Level-A fixed-trajectory physical
feasibility packet and do not attach learned-method labels to its motions.
Run that stronger gate with `--mode level_b`; it outputs
`LEVEL_B_METHOD_TRIALS_MAY_START` only when the action bridge is present.

After manually validating one low-speed fixed trajectory for each condition,
generate the Level-A randomized order with the three actual trajectory IDs:

```powershell
python scripts/generate_real_robot_level_a_schedule.py `
  --intact-trajectory-id <validated-id> --d2-trajectory-id <validated-id> `
  --d3-trajectory-id <validated-id> `
  --output data/real_robot/level_a_session_20260901.csv
```

Do not invent these IDs before validating the motions. The standard Push
analyzer now reports a separate `physical_feasibility_by_condition` table and a
formal Level-A gate (ten valid trials each for intact/D2/D3 plus raw-file checks),
whose claim boundary explicitly excludes learned-method superiority.

After the strict analyzer accepts the Level-A packet, generate its paper assets
directly from the JSON (never manually transcribe measurements):

```powershell
python scripts/build_real_robot_feasibility_assets.py `
  results/real_robot/push-summary.json `
  --figure paper/generated/real-robot-feasibility.pdf `
  --table paper/generated/real-robot-feasibility-table.tex
```

The figure and table are visibly scoped to physical feasibility and cannot be
generated when the summary contains no valid physical evidence.

Before analysis, prove that measurement entry did not alter the frozen trial
identity or remove a failed trial. Keep the blank frozen schedule and completed
log as separate files, then run:

```powershell
python scripts/audit_real_robot_schedule_completion.py `
  data/real_robot/level_a_schedule_frozen.csv `
  data/real_robot/level_a_trials_completed.csv `
  --output results/real_robot/schedule-completion-audit.json
```

This audit allows measurement/video/log fields to be filled but requires every
trial order, condition, position, method, and trajectory ID to remain identical.
It does not replace the separate validity and raw-file gate.

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

Generate and freeze the randomized block order before the first method trial:

```powershell
python scripts/generate_real_robot_push_schedule.py `
  --seed 20260901 --fault-pairs 10 --intact-pairs 5 `
  --methods nominal,global_matched `
  --output data/real_robot/push_trials.csv
```

Copy the emitted SHA-256 into the session manifest. If time has been formally
reserved for attribution, add `si_ipwm` to `--methods` before generation;
never append it after looking at nominal/global outcomes.

The repository already contains the primary two-method schedule at
`push_schedule_seed20260901.csv`: 25 paired blocks and 50 trials, comprising 5
intact, 10 D2, and 10 D3 pairs. Its frozen SHA-256 is
`79139bca3b61866643e00ef35d724cdd4185fb14a8f115faa942635f27f4510d`.
Use it unchanged or create a dated protocol deviation before trial 1; do not
silently shorten or reorder it after collection begins.

The analyzer labels fewer than ten complete reference/candidate pairs as `pilot`.
Ten or more pairs only changes the evidence level to `formal`; it does not imply
statistical significance or a positive result.

After strict analysis passes, build the paper-ready vector figure and generated
LaTeX table directly from the summary (never transcribe numbers manually):

```powershell
python scripts/build_real_robot_paper_assets.py `
  results/real_robot/push-summary.json `
  --figure paper/generated/real-robot-push.pdf `
  --table paper/generated/real-robot-push-table.tex
```

## Success definition

Freeze the success threshold before method labels are inspected. Always report
continuous terminal error, reach rate, contact rate, maximum lock error, aborts,
and failure codes even when a binary success rate is shown.

The frozen analyzer additionally reports relative endpoint-error reduction and
relative failure-rate reduction. These are predeclared descriptive effect sizes,
not substitute significance tests. Always show the absolute paired success
difference, its bootstrap interval, the reference/candidate failure rates, and
the counts of `candidate rescues reference failure` versus `candidate breaks
reference success`. If the reference has zero failures, relative failure-rate
reduction is undefined and is emitted as `null`, never as an infinite gain.

The two visual sources are fixed eye-to-hand cameras. Do not describe either as
eye-in-hand.

## Secondary fixed-pregrasp grasp feasibility

Grasp is deliberately secondary and does not train or evaluate a learned grasp
generator. Place the same cube in a marked pose, move to one frozen pregrasp,
close the gripper, lift vertically by the smallest safe repeatable distance,
and hold for three seconds. Run at most five intact/D2/D3 repetitions after the
Push packet is secure. Record every trial in `grasp_trials_template.csv` and
analyze it with:

```powershell
python scripts/analyze_real_robot_grasp.py data/real_robot/grasp_trials.csv `
  --require-files --output results/real_robot/grasp-feasibility-summary.json
```

This panel may support only reach/closure/retention feasibility. It cannot be
described as learned grasping, task-general recovery, or a method comparison.
