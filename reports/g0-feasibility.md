# G0 Feasibility Report

Status: **PASS WITH DOCUMENTED SCOPE DEVIATIONS - Reach baseline frozen**

## Scope decision

- Project control interface: J1-J5 (5 DoF); servo ID6 is gripper opening.
- Candidate lock failures: D2/J2, D3/J3, D4/J4.
- J6 policy: fixed tool/gripper assembly until real-arm inspection.
- Tasks: Reach first; Push only after Reach passes G0.

## Available evidence

- Analytic 5-DoF FK matches the simplified MuJoCo `ee` site in tests.
- Full seven-mesh GenkiArm visual model loads through the same 5-DoF API.
- Training and visual models now share the measured J1-J5 skeleton and TCP.
- D2/D3/D4 locked-joint reachability and common-target sampling are implemented.
- Topology and residual context are concatenated before world-model prediction.

## Required real-arm measurements

| Gate | Measurement | Result | Pass criterion |
|---|---|---:|---|
| Kinematics | FK endpoint error over 5 poses | PASS | Max measured component error 40 mm < 50 mm task tolerance |
| Reachability | Common region for intact + D2/D3/D4 | PASS | 1,601 common 3-cm voxels; 12 targets sampled |
| Lock safety | Current and temperature under D2/D3/D4 lock | PASS | 15-degree loaded-motion holds passed |
| Lock repeatability | Repeated D2/D3/D4 loaded holds | PASS | 78 cycles, 687 s, max 35 C |
| Emergency stop | Tested stop path | PASS | 102.7 ms command path; zero post-stop drift |
| Limits | Zero, direction, soft limits | PASS | All J1-J5 and gripper recorded |
| Safe speed | Conservative command profiles | PASS | Verified without overshoot |

Zero-pose validation errors were at most 1.2 degrees across J1-J5. The
measured mechanical ranges and conservative software ranges are recorded in
`hardware/safety_limits.yaml`; raw evidence is under
`hardware/calibration/raw/2026-08-10/session-03/`.

J5 requires the model-777 command profile `goal_speed_raw=0`,
`acceleration_raw=1`, and increments no larger than 10 degrees. A visible
30-degree clockwise/counterclockwise round trip passed with about 0.4 degrees
return error. `scripts/recover_j5.py` restores a non-moving position-mode
state without enabling torque.

The gripper passed a visible half-open to near-full-open round trip and uses
the same automatic-direction profile with increments capped at 300 ticks.

Calibrated simplified-model reachability was sampled with 50,000 joint
configurations per morphology. The intact/D2/D3/D4 intersection contains
1,601 occupied 3-cm voxels, and all 12 sampled common targets produced valid
dataset episodes. Tracked evidence: `results/final/g0-reachability-summary.json`.

The supplied STL files are millimetre CAD geometry. Their previous 0.002 mesh
scale doubled the arm; it is corrected to 0.001 without deforming any mesh.
Each visual link is attached to the measured 120/230/350/410 mm joint-axis
skeleton, and the visual and training TCP positions agree to numerical
precision over the checked poses. Render evidence:
`reports/g0-mesh-calibrated.png`.

Ten conservative real-arm poses now cover height and radial TCP displacement.
The first five were measured directly; P6-P10 were confirmed by the user as
matching the model predictions. The largest independently measured component
error remains 40 mm, below the plan's initial 50 mm Reach tolerance. Raw
evidence:
`hardware/calibration/raw/2026-08-10/session-03/24_fk_pose_validation.csv`.

The repeated loaded-hold test accumulated 78 D2/D3/D4 cycles over 687 seconds.
The clean final 240-second run had 0.79-degree maximum drift; the largest
observed J2 reversal backlash across all runs was 3.08 degrees. Temperature
never exceeded 35 C. J2 backlash must be represented in simulation
randomization and retained as a real-arm safety margin. Raw evidence is in the
three `25_lock_repeatability*.csv` run files.

## Full Gate Review

The calibrated URDF, URDF gap report, position/orientation reachability JSON and
plot, dynamics summary, safety boundaries, and task-scope decision are now
tracked. Position-only common reachability contains 2,591 25-mm voxels; the
tool-axis <=30-degree common region contains 324 voxels (12.5%). Reach is
retained, Push is an optional extension, and Pick is removed.

The ten-pose count is complete with P6-P10 explicitly marked
`pass_user_confirmed`; these five rows are confirmation-based rather than new
independent ruler readings. The independently measured poses satisfy the
50-mm Reach tolerance. J2-J5 synchronized step responses are
now recorded in `26_step_response.csv`, but their dynamics remain an empirical
safety trace rather than full inertial/friction identification. The tested
hardware has no independent J6 orientation actuator; ID6 is gripper opening.

Under the reduced position-only Reach scope, G0 passes. This does not authorize
claims about identified dynamics, orientation recovery, Push, or Pick.
