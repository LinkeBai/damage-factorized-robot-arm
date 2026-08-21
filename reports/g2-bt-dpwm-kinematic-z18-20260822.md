# BT-DPWM kinematic projection: Z18

Cached trajectories support an effective integration step of approximately
0.005 seconds. Semi-implicit integration leaves about 29.4% of the original
position-increment RMSE.

A parameter-free projection blends the learned next position with
`q + 0.005 * next_qvel`. On seed 7, four-domain overall improvement changes
from +2.12% at blend 0 to +2.51% at blend 0.75; pure integration gives +2.48%.
All have one regressing cell.

With blend 0.75 frozen, the three-seed/four-domain audit gives free -0.37%,
object +21.96%, overall +0.58%, and 6/12 regressions. Z18 is NO-GO. The
kinematic relation is valid but does not align the strong baseline's robot
errors across independently trained seeds.
