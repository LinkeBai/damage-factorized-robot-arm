# BT-DPWM reaction supervision and initialization: Z13--Z14

## Object-only upper bound

Using raw strong-baseline RMSEs, setting object error to zero while preserving
the baseline robot gives only +2.26% mean overall improvement across the 12
cells. The +5% gate therefore requires repeatable robot improvement.

## Z13: contact-supervised reaction

MuJoCo contact labels are used only during training to weight one-step joint
residual identification. Deployment still uses analytic geometry. On seed 7,
scale 0.25 gives free -0.73%, object +25.56%, overall +0.83%; larger scales
degrade monotonically. Object contact is not a sufficient joint-reaction label.

## Z14: deterministic physical feature basis

The physical adapter's first layer uses the same deterministic initialization
for every seed. Seed 7 has a broad passing interval: scale 0.30/0.40/0.50 gives
overall +5.14/+5.29/+5.00%. With scale 0.40 locked before replication, the full
audit gives free +0.84%, object +21.98%, overall +1.74%, and 4/12 regressions.

Both candidates are NO-GO. Adapter initialization is not the source of the
cross-seed failure. The remaining variance is tied to training-data/scaffold
residual inconsistency, so further scale, gate or seed-specific selection is not
justified.
