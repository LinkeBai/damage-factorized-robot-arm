# Calibrated GenkiArm native-training smoke test

## Purpose and evidence boundary

This smoke test verifies that the frozen training stack can consume trajectories
generated natively from `sim/assets/genkiarm_push.xml`. It is not part of the
five-seed result matrix and must not be cited as confirmation evidence. The
adapter smoke intentionally uses the old seed-27 base checkpoint and is named
accordingly to prevent it from being mistaken for an independently trained
seed-107 model.

## Changes required before the test

- The Warp collector now resolves `j1`--`j5`, `block_x`, and `block_y` by name
  instead of assuming an XML coordinate order.
- Context-encoder and adapter training accept an explicit `--xml` argument and
  write the source XML into `summary.json`.
- Base-model training accepts `--xml`; the resolved XML is embedded in every
  trajectory-cache key, preventing simplified-arm cache reuse.
- Smoke mode is explicitly tagged and reduces trajectories/epochs. It disables
  expensive goal-query collection only in smoke mode.

## Verified artifacts

| Stage | Output | Verification |
|---|---|---|
| context encoder | `runs/smoke_genkiarm_v2/context_seed107` | XML recorded; finite best validation loss 0.095236; checkpoint loads |
| adapter interface | `runs/smoke_genkiarm_v2/adapter_seed107_on_seed27_base` | XML and seed-27 base provenance recorded; both adapter checkpoints load |
| base world model | `runs/smoke_genkiarm_v2/base_seed107` | XML recorded; structured and baseline checkpoints load; complete smoke evaluation |

The base smoke result was object improvement +35.44%, free-arm improvement
+0.40%, and overall improvement +2.00%; its legacy aggregate gate returned
No-Go. These values are only a numerical/interface sanity check because the run
uses two trajectories and two epochs. They are neither positive nor negative
paper evidence.

The relevant regression suite passed 16/16 tests after the changes.

## Next action

Run the five fresh base-training seeds 107/117/127/137/147 with smoke mode off,
then train their GenkiArm adapters and context encoders. Only after all required
checkpoints exist may the frozen V2 evaluation matrix be executed.
