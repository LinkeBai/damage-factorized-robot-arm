# BT-DPWM shadow object context: Z19

After excluding weak-baseline experiments, the cell-level oracle over existing
strong-baseline candidates reaches free +4.45%, object +21.99%, overall +5.21%,
but still has two regressing cells. Existing mechanisms are complementary, yet
there is no deployable selector.

Z19 separates the object state used by the robot scaffold from the independent
object expert's output. A rank-8 shadow context head uses 1,164 parameters;
the full model has 338,074 parameters, 28 fewer than shared h136/240. The robot
is frozen, shadow context is trained for joint rollout fidelity, then the
independent object block is trained.

Seed 7 primary reaches free +4.71%, object +42.14%, overall +8.09%. Its
four-domain mean is free +2.35%, object +25.55%, overall +3.71%, with one
regression. Locked three-seed replication gives free +0.17%, object +21.96%,
overall +1.10%, and 4/12 regressions. Z19 is NO-GO: context isolation reduces
feedback drift but does not create repeatable robot improvement.
