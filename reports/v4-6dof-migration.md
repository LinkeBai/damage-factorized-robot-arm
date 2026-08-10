# V4 Six-DoF Simulation Migration

Date: 2026-08-07

Plan baseline: `PROJECT-PLAN-V4.md` (2026-08-06)

## Completed

- Restored J6 as the gripper assembly orientation axis in simple and mesh MJCF.
- Corrected the nominal chain so J4 is no longer co-located with J3.
- Unified action, mask, lock angle and joint mapping to six dimensions.
- Unified proprioceptive state to 12 dimensions (six qpos + six qvel).
- Added J1-J6/URDF/servo placeholder mapping in `hardware/joint_map.yaml`.
- Rebuilt FK and position-only IK against the six-joint MJCF.
- Converted residual physics profiles and G1 data generation to six joints.
- Added immutable G1 domain and provisional Reach target YAML splits with SHA256.
- Added 1000-step no-NaN, deterministic rollout and mapping tests.
- Added a minimal conditional RSSM prior/posterior and frozen CEM-MPC path.
- Added four prediction methods for the provisional G1 comparison.

## Superseded evidence

`results/final/g1-benchmark-20260807/` was generated using the earlier
five-joint interface. It remains an engineering pretest and is not valid G1
gate evidence under the new plan.

## Still blocked by G0

- Real J1-J6 servo IDs, zero positions, directions and units.
- Measured J6 role, gripper-open channel and TCP.
- Real soft limits, speed/current/temperature limits and emergency stop.
- Calibrated inertial, collision, backlash, compliance and latency ranges.
- Final common-reach target split replacing the provisional nominal split.
