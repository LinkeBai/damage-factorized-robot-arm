# GenkiArm mesh model

`arm_mesh.xml` is adapted from the senior project's `test4.xml` and uses the
provided `AA.stl` through `GG.stl` meshes.

The model exposes the same five-joint API as `../arm.xml`: `j1` through `j5`,
actuated by `m1` through `m5`. Servo 6 opens and closes the real gripper and
is kept outside the positioning-chain action.

Use:

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf sim\assets\genkiarm\arm_mesh.xml
```

The STL meshes are visual geometry. Their physical values are placeholders
and must not be presented as calibrated sim-to-real parameters.
