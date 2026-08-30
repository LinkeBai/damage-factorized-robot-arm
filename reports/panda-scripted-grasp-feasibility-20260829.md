# Panda physical Grasp feasibility and scripted baseline

## Scope

This experiment establishes that the second-arm task model supports a
repeatable contact-defined Grasp. It does **not** show that SI-IPWM improves
Grasp prediction or control. GenkiArm is excluded from formal Grasp metrics
until measured finger geometry and a calibrated gripper actuator are added.

## Protocol

- Model: official Franka Panda wrapper `sim/assets/panda_push_grasp.xml`.
- Controller: bounded inverse kinematics solved before execution, followed by
  native joint-position actuation through approach, descend, close and lift.
- Object: 50 mm cube with five deterministic XY perturbations sampled within
  +/-5 mm.
- Grasp physics: upstream dual fingers, equality constraint, tendon actuator,
  collision pads and MuJoCo contact dynamics.
- Prohibited shortcuts: no weld/equality attachment to the cube and no
  hand-written `grasped=True` state.
- Success: at least 50 mm physical lift and at least five simulation steps with
  simultaneous left- and right-finger contact.

## Result

The scripted baseline passed 5/5 trials. Mean lift was 0.142221 m. Every trial
recorded 1,151 bilateral-contact steps and approximately 1,178 total
finger-contact steps. The complete stage traces and raw trial rows are stored
in `runs/panda_scripted_grasp_baseline_v1/summary.json`.

This clears only the Grasp task-feasibility gate. A learned-method comparison,
fault intervention, action-ranking experiment and closed-loop SI-IPWM result
remain absent; the paper must not present this baseline as method evidence.
