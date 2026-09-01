# Real-robot onsite readiness audit — 2026-09-01

## Scope

This is a connection/readiness record, not task evidence. No robot motion was
commanded and no real Push trial was collected.

## Verified connections

- Overhead eye-to-hand camera: Daheng Imaging `MER2-230-168U3C`, serial
  `FDE23080341`, USB3 Vision. Galaxy SDK enumerated one device and a 1920x1200
  frame was acquired before GalaxyView took exclusive access.
- Horizontal eye-to-hand camera: `icspring camera`, Windows instance
  `USB\VID_32E6&PID_9211&MI_00\7&84A3818&0&0000`. DirectShow frame acquisition
  passed. Its field of view includes robot links and the block, but it should
  be widened enough to retain the complete gripper/contact region.
- `USB2.0 HD UVC WebCam` is not an experiment camera and must not be used as the
  overhead source.
- Serial converter: `USB-Enhanced-SERIAL CH343 (COM3)`, Windows status OK.

## Important protocol distinction

Windows `mode COM3` reported a stored/default 19200 8N1 configuration. The
repository's `ServoBus` opens the port explicitly at 1,000,000 baud. Therefore
19200 is not evidence of the robot protocol and must not be copied into the
experiment manifest.

## Read-only servo result

A read-only query was attempted at 1,000,000 baud for servo ID 1, register 56
(present position). It sent no goal position, torque-enable, speed, or motion
write. The servo did not respond before timeout. The run stopped immediately;
IDs 2--5 were not probed after the first timeout.

Possible causes include absent servo-bus power, an emergency-stop power break,
wrong bus connector, different IDs/protocol/baud, or a controller state not
represented by the historical repository configuration. None may be resolved
by guessing and sending writes.

## Current gate

`ROBOT_MOTION_NOT_AUTHORIZED`.

Before any low-speed motion, the onsite operator must verify main servo power,
emergency-stop operation, cable/bus connection, and the original controller
configuration. Then run `scripts/probe_servo_bus_readonly.py`; all five expected
servos must return position and temperature. Passing the read-only probe still
does not authorize formal trials: joint directions, measured limits, stop
behavior, camera calibration/synchronization, and three fixed trajectories
must subsequently pass their own gates.

GalaxyView was observed holding exclusive access to the Daheng camera. This is
acceptable while adjusting exposure, but it must be closed before repository
capture/calibration code opens the device.
