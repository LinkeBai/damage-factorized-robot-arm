# Real-robot action-interface audit

## Verdict

**Learned-method hardware comparison is not yet authorized.** The simulation
world models consume MuJoCo `motor` controls: generalized force commands applied
at a 0.005 s physics step and held for ten steps per segment. The physical arm
accepts servo goal-position commands in raw ticks. A shared action label is not
evidence that these two interfaces have the same transition meaning.

The repository currently contains measured joint zeros, directions, conservative
position limits, a 5 deg/s speed limit, lock-repeatability logs, and position
step responses. It does not contain an identified mapping from a simulated motor
sequence to a time-indexed physical position trajectory, nor a frozen action
library generated through such a mapping. Therefore the predeclared
`nominal`/`global_matched` schedule is ready as an order template but cannot yet
be interpreted as a learned-policy comparison.

## Two-level hardware evidence

### Level A: physical mechanism/feasibility (authorized after ordinary safety preflight)

- verify intact, D2, and D3 lock error under low speed;
- use one manually verified, fixed position trajectory per condition;
- record reach, contact, terminal block displacement, aborts, both fixed
  eye-to-hand videos, and joint feedback;
- run repeated Push and the optional fixed-pregrasp short lift;
- claim only hardware feasibility, constraint validity, and the six-stage
  measurement protocol—not model superiority or sim-to-real control.

### Level B: learned-method comparison (not authorized yet)

Before the nominal/global schedule can be used as method evidence, freeze and
archive all of the following:

1. a mapping specification from each simulated motor segment to a physical
   joint-position trajectory, including units, duration, interpolation, clipping,
   and locked-joint handling;
2. low-amplitude intact/D2/D3 validation logs showing direction agreement,
   limit compliance, lock drift below 3.5 deg, and no stop event;
3. one common candidate library used by both models, with its SHA-256 recorded
   before method outcomes are inspected;
4. the model-selected candidate ID for every pair, so method labels are tied to
   an auditable computation rather than a hand-assigned motion;
5. a deviation record if the hardware candidate count or horizon differs from
   the frozen 128-by-50 simulation protocol.

## Immediate on-site sequence

1. Photograph and calibrate the setup; verify emergency stop and feedback.
2. Complete Level A with low-amplitude fixed trajectories first. Preserve every
   abort and failed contact.
3. Collect action-interface identification probes only if Level A is safe.
4. Do not start the nominal/global formal comparison unless the action bridge
   and common library are frozen and independently replayable.
5. If Level B cannot be completed, retain Level A honestly as a feasibility
   panel and keep the manuscript's current simulation-only method claim.

This separation is mandatory because a hardware result with unverified action
semantics would add apparent breadth while weakening technical correctness.
