# Original 5-DoF real-robot Push packet

This directory contains only schemas and templates until measurements are
collected. Blank cells are not evidence and must never be imputed.

Before the first method trial, copy `session_manifest_template.yaml` to a new
session-specific manifest, fill every field, compute the randomized schedule
hash, and set the freeze record. The binary success threshold is fixed at 30 mm
to match the simulation protocol; changing it requires a dated deviation and
both old- and new-threshold results.

Run the hard preflight immediately before trial 1. Exit code 0 and
`FORMAL_TRIALS_MAY_START` are both required; `--skip-path-existence` is only for
schema tests and is forbidden during the real session:

```powershell
python scripts/audit_real_robot_preflight.py data/real_robot/session_20260901.yaml `
  --output results/real_robot/preflight-audit.json
```

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
