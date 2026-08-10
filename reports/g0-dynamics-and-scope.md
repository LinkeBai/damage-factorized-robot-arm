# G0 Dynamics, Safety Boundaries, and Task Scope

Date: 2026-08-10

## Derived Dynamic Evidence

Three zero reads were identical for all six servos. The synchronized J2-J5
step-response run is recorded in `26_step_response.csv`. J2-J4 began responding
after approximately 0.5-0.7 s and entered a 1-degree error band in 2.2-2.4 s;
J5 began responding after 0.1-0.3 s and entered the band in about 0.6 s.
Final step errors were below 1.5 degrees for all measured phases.

Visible round-trip errors
were J1 0.35 deg, J2 0.53 deg, J3 0.70 deg, J4 0.53 deg, and J5 0.40 deg.
The sampled J1 low-speed move first changed by the 393 ms sample and settled by
1,522 ms; the logger did not record a synchronized command edge, so this is a
bounded response trace rather than a formal latency estimate.

Loaded D2/D3/D4 holds had zero encoder-tick drift over five seconds. Across
78 repeated cycles and 687 seconds, the final run maximum drift was 0.79 deg,
the largest J2 reversal event was 3.08 deg, and temperature never exceeded
35 C. Current telemetry was readable in raw units (maximum observed 1), but
the available data do not establish a defensible raw-to-mA conversion. The new
step run reached 39 C maximum and observed current raw values 0-2.

## Safety Boundaries

- Base/J1: +/-85 deg, 5 deg/s, stop on feedback loss.
- Intermediate/J2-J4: measured software ranges; 5 deg/s; loaded lock only;
  abort at 50 C, current raw 400, or 3.5 deg locked displacement.
- Wrist/J5: +/-90 deg; speed register zero for automatic direction;
  acceleration raw 1; increments no larger than 10 deg.
- Gripper opening/ID6: raw 1050-2185; increments no larger than 300 ticks.
- Emergency stop: sequential torque disable IDs 1-6, measured command path
  102.7 ms with zero 500 ms post-stop drift.

## Task Decision

- **Reach: retain** as the primary task. Position-only common reachability is
  broad and the deterministic controller reaches all frozen targets.
- **Push: conditional extension only.** The orientation-constrained common
  region is 12.5% of the position-only region; contact dynamics are not
  physically identified.
- **Pick: remove from the main scope.** There is no independent J6 orientation
  actuator on the tested arm, finger geometry is not modelled, and camera-based
  grasp evaluation is outside the current evidence.

No unmeasured physical quantity is replaced by an invented point estimate.
