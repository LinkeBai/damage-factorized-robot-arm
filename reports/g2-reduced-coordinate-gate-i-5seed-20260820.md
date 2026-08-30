# Gate I Five-Seed Audit: Reduced-Coordinate Graph World Model

**Decision: NO-GO as a stable main method**

RC-GWM removes locked joints from the transition graph, reconnects nearest
free neighbors, and predicts only the reduced free-coordinate dynamics. It
achieves exact zero constraint violation by construction and is more
structurally novel than soft topology conditioning or post-hoc projection.

## Primary `D3__mixed_composition` results

| seed | object regression vs matched | free-arm regression vs matched | violation | gate |
|---:|---:|---:|---:|---|
| 7 | -50.00% | -1.22% | 0 | PASS |
| 17 | +136.02% | +84.00% | 0 | NO-GO |
| 27 | -19.84% | +37.28% | 0 | NO-GO |
| 37 | +46.36% | +41.89% | 0 | NO-GO |
| 47 | -43.09% | -3.88% | 0 | PASS |

Only **2/5 seeds** satisfy the pre-registered 5% fidelity gate. The method's
exact constraint mechanism is stable, but its learned free-arm dynamics are
strongly seed-sensitive. Object prediction often improves while free-arm
prediction collapses, indicating an optimization/objective conflict rather
than a reliable architectural gain.

## Decision

Do not continue RC-GWM to further seeds or present it as a stable main method.
Retain it as a technically meaningful negative result: coordinate reduction
solves feasibility, but the current shared graph/object objective does not
preserve free-arm fidelity. The next legitimate experiment would require a
newly specified stability intervention (for example, separate free-arm/object
heads with gradient-conflict control), not unregistered hyperparameter tuning.

The paper's stable primary route remains the existing five-seed ensemble
uncertainty and selective-prediction evidence.

## Artifacts

- `config/experiment/g2_reduced_coordinate_gate_i_5seed_v1.yaml`
- `src/robotarm/models/reduced_coordinate_graph.py`
- `runs/g2_reduced_coordinate_gate_i/seed{7,17,27,37,47}_v1/summary.json`
