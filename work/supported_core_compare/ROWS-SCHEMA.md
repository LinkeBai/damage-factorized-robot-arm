# Supported core comparison: evaluation row contract

All paths below are relative to `runs/lockpusher_supported_core_20260913`.
The protocol is frozen independently. Neither evaluator can access test models
before `training-complete.json` verifies all nine selections and the source and
implementation manifests match. Each final result directory is atomically
committed; an `.incomplete-*` directory never counts as a completed run.

## Prediction

`prediction/{method}/seed{seed}/{rows.npz,complete.json}`, 6,000 rows per fit.
Methods are `ipwm`, `carrier`, `global`; seeds are 7, 17, 27.

Identity fields: `reset_id`, `initial_state_sha256`, `locked_joint` (0-based),
`profile`, `free_joint_mask[n,5]`. H10/H25/H50 share the same trajectories.

For each H, prefix every following key with `h{H}_`:

| Key | Shape | Meaning |
|---|---|---|
| object_xy_squared_m2 | n,2 | Squared predicted-minus-measured XY error, meters squared |
| object_velocity_squared_m2_s2 | n,2 | Squared planar velocity error |
| joint_q_squared_rad2 | n,5 | Per-joint squared position error, including locked joint |
| joint_velocity_squared_rad2_s2 | n,5 | Per-joint squared velocity error |
| free_q_mean_squared_rad2 | n | Sum over four free joints divided by four |
| free_velocity_mean_squared_rad2_s2 | n | Sum over four free joints divided by four |
| pusher_xy_squared_m2 | n,2 | Squared analytic FK pusher XY error |
| lock_position_abs_rad | n | Predicted locked q minus the original diagnosed angle, absolute |
| lock_velocity_abs_rad_s | n | Predicted locked velocity, absolute |
| reference_q_abs_rad | n,5 | Predicted q minus paired reference q, absolute |
| reference_velocity_abs_rad_s | n,5 | Predicted velocity minus paired reference velocity, absolute |
| reference_pusher_xy_abs_m | n,2 | Predicted versus paired-reference FK pusher difference, absolute |
| predicted_state14 | n,14 | Returned forecast |
| reference_state14 | n,14 | Independently rolled paired carrier |
| truth_state14 | n,14 | Measured test state |

The paired reference is the **selected adapted no-object-residual carrier of
the same fitting seed**, rolled from the same initial state and raw commands
with its own recurrent state. Its exact checkpoint hash is in `complete.json`.
It is not claimed to be an independently trained robot or an isolation ablation.
The robot weights are frozen by the fitting protocol; global correction can
still change its returned robot block.

`reference_q_all_steps_max_abs_rad` and
`reference_velocity_all_steps_max_abs_rad_s` are per-trajectory maxima across
all 50 steps and all five joints, with position and velocity kept separate.
`lock_position_all_steps_max_abs_rad` and
`lock_velocity_all_steps_max_abs_rad_s` are the corresponding diagnosed-lock
maxima over every step.

Position/pusher RMSE in mm is `1000*sqrt(mean(squared_error_array))`, averaging
both trajectories and the two coordinates. Velocity uses the same formula
without the factor 1000 and is in m/s. Free q/v RMSE is the square root of the
mean per-trajectory free-joint MSE. No unit-mixed aggregate is provided.
The pusher is `contact_geometry.pusher_reference_point(q)`, computed on predicted
versus observed joint positions; it is a common FK diagnostic, not separately
measured hardware tip position. Full z/vz is retained in source data, not the
14-D learned input.

## Planning

`planning/{method}/seed{seed}/D{1..5}-B{0..1}/{rows.npz,resets.json,complete.json}`.
Each cell has n=120 shared problems. Seed 7/17/27 uses candidate repeat 0/1/2.
Five lock cells and two distance bands give 1,200 independent problem identities;
fits/methods reuse them and do not multiply independent starts.

| Key | Shape | Meaning |
|---|---|---|
| reset_id, initial_state_sha256, profile | n | Reset provenance |
| locked_joint, band | n | 0-based lock and distance band |
| goal_xy_m | n,2 | Assigned planar world goal |
| initial_distance_m | n | e0 = norm(initial XY - goal) |
| terminal_distance_m | n | eT = norm(final executed XY - goal) |
| progress_m | n | e0 - eT; negative values retained |
| relative_terminal_error | n | eT / e0, computed per problem |
| success_at_30mm | n | eT < .03; no early stopping |
| object_net_displacement_m | n | Norm(final XY - initial XY), not goal progress |
| states | n,51,14 | Initial state and every 5 ms observation after mj_forward; 0.25 s total |
| full_qpos, full_qvel | n,51,8 | Complete supported model q/v including block z/vz |
| time_s | n,51 | MuJoCo time |
| controls | n,50,5 | Executed raw motor command during every 5 ms observation interval |
| diagnostic_metrics | n,51,K | `data.METRIC_NAMES`, also saved as diagnostic_metric_names |
| substep_diagnostic_metrics | n,50,S,K | Full diagnostics after every internal physics step; S = .005 / integration timestep |
| substep_time_s | n,50,S | Time after every internal physics step |
| substep_full_qpos, substep_full_qvel | n,50,S,8 | Full simulated q/v after every internal physics step |
| contacts | events,8 | trajectory_index, observation_step, internal_substep, geom1_id, geom2_id, distance_m, normal_force_N, tangent_force_norm_N |
| first_tool_object_contact_step | n | First 1-based observation interval with positive-force tool/pusher-to-block contact, -1 if absent |
| first_tool_object_contact_substep | n | First 1-based internal substep in that observation interval, -1 if absent |
| tool_object_contact_step_count | n | Number of observation intervals containing at least one such contact |
| tool_object_contact_internal_step_count | n | Number of internal physics transitions with that contact |
| max_object_displacement_before_tool_contact_m | n | Largest XY displacement before any such contact |
| chosen_indices | n,5 | Selected candidate at each replan |
| chosen_segment_commands | n,5,5,5 | Selected raw-command sequence: replan, segment, joint |
| predicted_candidate_terminal_distance_m | n,5,128 | Model-predicted terminal goal costs for every candidate/replan |
| predicted_candidate_terminal_xy_m | n,5,128,2 | Corresponding terminal object predictions |
| candidate_array_sha256 | 5 | Complete candidate array hash for each replan |
| candidate_rng_entropy | 5,6 | SeedSequence identities for regenerating candidate arrays |
| goal_angle_world_rad, goal_direction_offset_rad | n | World direction and perturbation from initial pusher-to-object direction |
| goal_base_direction_unit_xy | n,2 | Center direction before the ±pi/4 perturbation |

`contacts` are captured after **each internal** `mj_step`, before `mj_forward`,
so they describe the solver contacts which generated that transition. Both
observation and internal-substep columns are 1-based; trajectory_index is
0-based. Diagnostics are captured after `mj_forward` at every internal step.
Dynamic extrema must be computed from the initial diagnostic frame **plus all
substep diagnostics**, not just the 51 observation frames, which can miss
between-frame penetration or joint-limit peaks. Complete internal q/v is also
retained. `common.step` determines the final validated integration timestep;
this interface does not choose physical parameters or authorize a failed gate.
All contact rows are retained; only contact classification applies 1e-9 N.
`complete.json` contains geom names by ID, joint order, diagnostics/contacts
column names, reset hash, model/source identities, and runtime components.

Every problem owns its own MuJoCo model and MjData. The equality target and
active equality are checked at the observation boundary and every internal
callback. Raw control stays fixed within each 5 ms observation interval. Only
the selected first ten-observation command segment (50 ms) is executed; the
next replan sees that method's realized state. All five replans execute even
after entering the goal region. Extra integration substeps do not change the
50-observation forecast, 50-observation episode, 128-candidate budget, five
segments or five replans. The evaluator checks 0.25 s final elapsed time.

Goals use the initial **pusher geom center to block center** direction, then
uniform ±pi/4, with radius uniform in .04–.065 or .065–.09 m. Goal RNG is
`[91362026,505,lock,profile_id,band,problem_index]`; reset index is
`band*120+problem_index`, profile is `problem_index%4`. Candidate RNG is
`[91362026,606,repeat,lock,band,replan]`. Candidates are uniform raw commands in
[-.8,.8], with the locked coordinate zero. No feasibility gate or true-cost
oracle was added.

`run_development_smoke(model, ...)` uses only four **development** resets and
eight candidates in a unique `smoke/planning/` directory. It cannot create a
formal complete record or read the test dataset/selected checkpoints.
