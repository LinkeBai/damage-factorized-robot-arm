# ID3 Safety Event

- Date: 2026-08-10
- Event: ID3 probe moved downward; return/upward motion did not complete; after
  torque release the supported link dropped under gravity.
- Immediate state: no further motion commands; ID3 torque was disabled by the
  test cleanup.
- Interpretation: positive direction and gravity behavior are not yet frozen.
  The failed return may be a mechanical or load-related issue; the drop after
  torque release must not be treated as a normal calibration result.
- Required before resuming: power-off inspection, independent mechanical support,
  and a return/hold procedure that verifies present position before torque release.
- Safety status: G0 motion testing paused for ID3 pending inspection.
