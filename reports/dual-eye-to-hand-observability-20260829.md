# Dual eye-to-hand observability under extrinsic perturbations

## Protocol

Both fixed eye-to-hand cameras are evaluated on calibrated GenkiArm and the
official Panda task wrapper. MuJoCo geometry segmentation is used only to audit
whether the task object remains observable; it is not used as an input to the
state-based SI-IPWM and is not a learned vision result.

The frozen perturbation set contains nominal extrinsics, +/-20 mm translations
along each world axis, and +/-3 degree yaw. Each of the nine conditions is
applied to both cameras on both robots at 320x240 resolution. Visibility
requires at least 20 object pixels in each camera.

## Result

Both cameras retained object visibility in 18/18 robot-by-perturbation
conditions. The minimum object area over all individual camera frames was 224
pixels. The maximum object-centroid displacement from nominal was 13.34 pixels;
the largest shifts came from the +/-3 degree yaw conditions.

Raw per-camera areas, centroids and shifts are in
`runs/dual_eye_to_hand_observability_v1/summary.json`.

## Claim boundary

This passes a camera-placement/observability feasibility gate only. The current
SI-IPWM receives simulator state rather than RGB observations, so these results
cannot support claims of visual world-model accuracy, calibration robustness,
multi-view fusion, or sim-to-real perception. Those claims require an explicit
vision estimator and end-to-end error propagation into prediction and control.
