# V6 Hybrid Gate

Status: **pass_with_gain**

The new method is stable, but stability alone is not evidence that the world model adds control value.

| Method | N | Success | Mean steps | Mean final error |
|---|---:|---:|---:|---:|
| ik_pd | 8 | 8/8 | 72.9 | 49.0 mm |
| jacobian_residual | 8 | 8/8 | 76.1 | 48.8 mm |
| worldmodel_hybrid_k0 | 24 | 24/24 | 72.1 | 48.7 mm |
| worldmodel_hybrid_k5 | 24 | 24/24 | 72.2 | 48.7 mm |

Hybrid stability: **PASS**.
World-model independent gain: **PASS**.

The current evidence supports a safe model-guided hybrid controller. It does not support claiming that the world model itself improves the verified IK/PD controller.
