# BT-DPWM scaffold and residual audit: Z22--Z30

Date: 2026-08-22. Comparator: shared h136/240, 338,102 parameters. Frozen
objective: three seeds by four test domains, positive mean free/object/overall,
at least +5% mean overall, and at most one regressing cell.

## Scaffold experiments

| Gate | Change | Seed-7 primary object | free | overall | Decision |
|---|---|---:|---:|---:|---|
| Z22 | last-80 SWA scaffold | +42.07% | -84.11% | -74.72% | NO-GO |
| Z23 | final/EMA/validation checkpoint selection | +42.03% | -93.50% | -83.53% | NO-GO |
| Z24 | analytic contact-gated context | +42.04% | -84.10% | -74.71% | NO-GO |
| Z25 | topology-conditioned scaffold from epoch 1 | +42.49% | -128.50% | -116.37% | NO-GO |
| Z26 | end-to-end semi-implicit scaffold | +42.04% | -92.24% | -82.35% | NO-GO |

Validation loss improved as far as 0.00680 in Z26, yet held-out D3 became much
worse. Scaffold selection and structural training loss are therefore invalid
proxies for the required topology OOD gate.

Z24 also exposed a geometry bug: planar distance treated vertical pusher passes
as contact. Full 3-D capsule/box separation removed these false positives. At
the old -5 mm threshold precision became 1.0 but recall was too low; train and
validation selected a 0 mm boundary. Frozen D3 contact F1 was 0.62, insufficient
to rescue the robot rollout. The corrected geometry remains tested infrastructure,
not evidence that Z24 passed.

## Frozen-scaffold residual experiments

Z27 adds one joint-shared linear map from ten physical features to q/qvel
correction: 22 parameters, zero initialized, closed-form ridge fit. Ordinary
validation selected lambda 1 and produced primary overall -49.60%. Z28 selected
lambda by leave-one-topology-out folds over intact/D2/D4; lambda 1000 still
produced -45.26%. Both are NO-GO.

Z29 transferred the identical development-seed physical reaction adapter to all
three frozen scaffolds. The strict 12-cell audit gave free -4.69%, object
+21.96%, overall -3.63%, and 8 regressions. Z30's best zero-parameter delta
contraction (q scale 0.90) reached only +2.52% overall on seed 7 four-domain
development, versus +2.12% at scale 1.0.

## Frozen decision

Do not retrain the h136 robot scaffold on the present leave-D3-out split. Do not
continue additive residual, scale, threshold, or validation-checkpoint sweeps.
The next admissible mechanism must be a predeclared deployable state-level
confidence/selection rule with zero correction as a safe path. Z4 remains the
strongest stable base; no Z22--Z30 result satisfies the final objective.

## Z31: state-level relative trust region

The existing rank-8 reaction is clipped per joint so its 2-D q/qvel correction
cannot exceed a fixed fraction of the frozen scaffold's own transition norm.
This adds no parameters and clip zero exactly recovers Z4. Seed-7 development
rose monotonically from +2.12% overall at clip 0 to +5.07% at clip 0.8, with
one regressing domain. After freezing 0.8, the 12-cell replication yielded free
-0.34%, object +21.99%, overall +0.60%, and four regressions. Z31 is NO-GO.

The failure exhausts the predeclared state-level safety path: magnitude control
cannot repair a correction whose direction is wrong on seed 27. Further progress
requires new training information or online calibration, not another test-driven
threshold sweep.
