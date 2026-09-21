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

After manually validating one low-speed fixed trajectory per condition, freeze
the common physical reset as position `A` and generate the Level-A order. The
default table contains 30 mandatory primary rows plus 10 preregistered reserve
rows per condition. Reserve rows are not optional cherry-picks: after all
primary rows, consume only each condition's frozen `reserve_rank` prefix until
that condition reaches ten valid, non-aborted trials.

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

Store the actual time-indexed joint-position waypoints in a trajectory-library
CSV with columns `trajectory_id,condition,waypoint_index,time_s,j1,...,j5`, then
audit it against the frozen schedule before execution:

```powershell
python scripts/audit_level_a_trajectory_library.py `
  data/real_robot/level_a_trajectory_library.csv `
  data/real_robot/level_a_schedule_frozen.csv `
  --output results/real_robot/trajectory-library-audit.json
```

The audit requires every scheduled ID to exist, measured joint limits, at most
5 deg/s, contiguous times, and constant J2/J3 commands under D2/D3. Record the
library path, SHA-256, and PASS audit path in the session manifest.

### Raw-tick teaching and single-trial execution

The hardware runner consumes the same operator-labelled library with the five
additional authoritative columns `j1_raw,...,j5_raw`.  The read-only teaching
helper appends both raw feedback and the corresponding `j1,...,j5` radians, so
one CSV can pass the existing Level-A audit and drive the raw-tick executor.
Nothing infers a trajectory ID, condition, index, or time:

```powershell
python scripts/capture_real_push_waypoint.py `
  --output <operator-waypoint-library.csv> `
  --trajectory-id <operator-validated-id> --condition intact `
  --waypoint-index 0 --time-s 0 --port COM3
```

Repeat with explicitly chosen contiguous indices and strictly increasing times.
The helper only reads STS present-position register 56; it never writes a
register or enables torque.  Do not place a block in the motion corridor while
teaching or validating a trajectory.

Before any hardware access, run the executor without `--execute`.  This checks
raw/radian agreement, raw limits, waypoint order, the 5 deg/s bound, the D2 J2
or D3 J3 constant target, interpolation, and the frozen camera-settings file:

```powershell
python scripts/run_real_push_fixed_trajectory.py `
  --waypoints <operator-waypoint-library.csv> `
  --trajectory-id <operator-validated-id> --condition intact
```

The output must say `DRY_RUN_VALIDATED_NO_HARDWARE_ACCESSED`.  That message is
not a trial result.  Only after the trajectory-library audit, session preflight,
camera exclusivity check, physical start-pose check, supported-arm check, and
tested E-stop may the operator add all of the following:

```powershell
  --execute --trial-id <schedule-trial-id> --port COM3 `
  --acknowledge-risk I_HAVE_CLEARED_WORKSPACE_SUPPORTED_ARM_AND_TESTED_ESTOP
```

The runner refuses to overwrite a trial directory.  It records native,
unannotated MJPG video from Daheng SN `FDE23080341` and DirectShow index 1,
`frame_timestamps.csv` with per-frame host grab intervals and Daheng hardware
timestamps when available, `commands.csv`, and a flushed five-servo telemetry
CSV containing measured/target raw ticks, voltage, temperature, and signed raw
current.  Camera/feedback loss, undervoltage, current, temperature, measured
limit, or lock-drift violations request torque-off for IDs 1-6 before cleanup.
Because STS writes have no acknowledgement packet, the runner first verifies
torque-off on IDs 1-6, then configures all five goal/acceleration/speed
registers while torque remains off and before either camera starts.  Static
phases always send the complete write batch, settle for 100 ms, and only then
read registers; confirmed mismatches alone receive bounded targeted rewrites,
while persistent read timeout fails closed.  After camera readiness and
pre-roll, the runner reads a fresh present-position snapshot, batch-seeds those
five latest positions, and enables all five torques only in a final independent
batch with the same settle/readback policy.  Every batch and correction is
retained in `run_manifest.json`.
High-frequency one-tick trajectory events write only joints whose target
changed; reading after every microstep is forbidden because it overloads the
serial bus.  Instead, all five goal registers are checked after at most ten
changed-target dispatches or 0.5 seconds, whichever comes first, and once more
unconditionally after the final event.  Runtime reads use two attempts with a
10 ms retry interval, within the configured 250 ms communication timeout; an
axis that remains unreadable fails immediately without correction writes or
later-axis reads.  A confirmed mismatch is repaired by at most three rounds of
rewriting only the mismatched axes, settling for 50 ms, and checking the goal
registers again.  A successful repair is retained as
`CORRECTED_AFTER_RETRY`; an unresolved mismatch aborts the run.  The original
mismatch, every correction write/read, policy, and validation outcome are
stored in the run manifest.

For D2/D3, torque is enabled only after every axis goal is seeded from that
axis's present position.  A separate `start_alignment` then moves at no more
than 5 deg/s to the frozen first waypoint.  Damage becomes active only after
alignment goal readback and locked-axis feedback are within the 3.5 deg gate;
the locked target cannot change during `fixed_trajectory`.  Alignment is logged
but is explicitly excluded from the task-trial motion count.

An abort performs an immediate all-ID torque-off sweep and `finally` performs a
second independent sweep; every write attempt, read value, timeout, still-on
ID, and uncertain ID is retained in `run_manifest.json`.  A capture must not be
treated as safely closed when its manifest says `NOT_VERIFIED_OFF` or marks the
latest torque readback uncertain.
Normal completion is labelled `ACQUISITION_COMPLETE_UNASSESSED`; the runner
never assigns reach, contact, success, or a learned-method label.

After the action-library audit passes and all real identifiers/paths are known,
use `scripts/prepare_real_robot_level_a_session.py --help` to generate the
session manifest. It reads the measured 5 deg/s and 3.5 deg lock-drift limits,
hashes the frozen schedule/library plus the current read-only servo-readiness
and camera-synchronization PASS audits, writes the actual paths, disables
learned-method claims, and records the freeze time. Do not hand-edit hashes.

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

This audit requires every primary row, preserves every abort, and allows only
the exact preregistered reserve prefix needed to reach ten valid trials per
condition. It also enforces `success == (endpoint_error_m <= 0.03)` and rejects
a non-aborted lock error over 3.5 degrees. It does not replace the separate
raw-file gate.

When collection is complete, run the entire Level-A evidence chain with one
fail-fast command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_real_robot_level_a_pipeline.ps1 `
  -Manifest data/real_robot/session_20260901.yaml `
  -FrozenSchedule data/real_robot/level_a_schedule_frozen.csv `
  -TrajectoryLibrary data/real_robot/level_a_trajectory_library.csv `
  -CompletedLog data/real_robot/level_a_trials_completed.csv
```

It stops at the first failed gate and produces paper assets only after trajectory
safety, preflight, schedule identity, measurement validity, and all raw files
have passed. Do not
run the asset builder separately to bypass an earlier failure.

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
