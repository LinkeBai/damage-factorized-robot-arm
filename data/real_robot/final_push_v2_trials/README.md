# Final push trials (V2, 2026-09-05)

Logs for the 29 recorded hardware push trials, including the 18 formal trials
listed in `../final_push_formal_membership_v4.csv` (intact arm and J1–J5 locks,
three trials each). Summary results are in
`reports/ipwm_final_submission_archive_ready_20260905/formal_results.csv`.

## Contents of each trial folder

| File | Content |
|---|---|
| `run_manifest.json` | Trial configuration, image goal, and per-cycle replanning records (including `candidate_bank_sha256`) |
| `live_task_gate.json` | Object detections and the goal-acceptance gate |
| `frame_timestamps.csv` | Capture timestamps for both cameras (frame-aligned) |
| `servo_telemetry.csv` | Joint targets and encoder feedback |
| `commands.csv` | Commands sent to the servos |
| `timing_summary.json`, `packet_audit.json`, `ipwm_closed_loop_evidence.json` | Timing, communication, and closed-loop audits |
| `ipwm_replan_cycles/cycle_XX.npz` | Per-cycle observation, candidate scores, selection eligibility, selected index and references, fault mask, and lock angles |

## Not included

These files did not fit the repository and were not kept:

- Raw camera videos (`daheng_FDE23080341_raw.avi`, `directshow_index1_raw.avi`)
- The full candidate trajectory bank per cycle (`candidate_references`,
  10,000 x 50 x 5). Its SHA-256 remains in `run_manifest.json`
  (`candidate_bank_sha256`).
- The per-trial `source_snapshot/` code copy. The code is tracked in this repository.

All included arrays were verified to match the originals exactly
(identical values and dtypes).
