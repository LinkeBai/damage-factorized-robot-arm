# URDF Gap Report

Date: 2026-08-10

## Source Audit

The upstream `genkiarm.urdf` defines six revolute joints and references
`AAA.stl` through `FFF.stl`. It has visual geometry only: no collision,
inertial, dynamics, transmissions, actuator mapping, or calibrated limits.
Its mesh names do not match the supplied `AA.stl` through `GG.stl` files.

The physical bus inspection found five positioning joints (IDs 1-5) and one
gripper-opening actuator (ID 6). It did not find an independent sixth
gripper-orientation joint. Therefore upstream `Rotation6` is not retained as
an actuated kinematic joint in the calibrated model.

## Calibrated Replacement

`sim/assets/genkiarm_calibrated.urdf` provides:

- measured J1-J5 origins, axes, zero convention, and safe limits;
- fixed tool and TCP frames;
- valid paths to the supplied seven STL meshes at millimetre scale 0.001;
- conservative primitive collision geometry;
- explicit placeholder inertials and dynamics so missing values are visible.

## Remaining Non-Truth Fields

- Link mass, center of mass, and inertia are placeholders.
- Damping, friction, effort, and velocity are conservative simulation values,
  not identified physical parameters.
- TCP lateral offsets `y=-0.0132 m` and `x=0.020 m` originate from upstream CAD
  and remain less certain than the measured 520 mm vertical height.
- Finger geometry and gripper opening are not represented as URDF joints; ID 6
  remains an independent hardware command channel.

The calibrated URDF is suitable for kinematics, visualization, and approximate
collision layout. It must not be cited as identified dynamics ground truth.
