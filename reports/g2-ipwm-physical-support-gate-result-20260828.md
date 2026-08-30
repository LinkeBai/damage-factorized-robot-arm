# IPWM observable physical-support gate result

## Verdict

**Object-spectrum gate: partial pass. Free-state safety gate: fail.**

The seed27 development audit exposed topology-only over-routing. A deployable
dual router was frozen before confirmation: enable the intervention residual
only when the L2 norm of the observable Z65 K25 context posterior is at least
1.2; otherwise use the no-intervention matched-adapter fallback. The router
uses no residual name, simulator parameter, test error, or domain label.

## Confirmation on seeds37/47

Across seven D3 physics conditions and H10/H25/H50, there are 42 confirmation
cells:

- 31/42 have nonnegative object improvement;
- 42/42 stay within a 2% object non-regression band;
- the worst object change is -1.893%;
- all locked-coordinate violations remain zero;
- the worst free-state regression is 24.456% (seed47, D3 high-damping, H50).

The fallback is effective for nominal, weak-motor, delay-1, and noisy-deadband.
Its small negative object cells are all within 2%. Full intervention remains
strong for mixed-composition and mixed-unseen. High-damping is correctly routed
to the full path for object prediction, but seed47 shows that object gain can
coexist with a large free-state cost.

## Interpretation

This router fixes the newly revealed object-spectrum brittleness without
retraining and confirms that observable physical context can distinguish when
the intervention path is useful. It does **not** solve free-state protection.
Therefore it can support a selective object-adaptation claim but cannot support
full-state non-regression or learned-control safety.

The next router must add a deployment-observable free-state validation signal
on held-out K25 support. The frozen candidate is accepted only if the full path
does not degrade free-state support error beyond the fallback tolerance. The
threshold for that support check must be developed separately and confirmed on
new evaluation seeds; the 24.456% cell cannot be used both to tune and confirm.

## Artifacts

- protocol: `reports/g2-ipwm-d3-physics-spectrum-protocol-20260828.md`;
- router config: `config/experiment/g2_ipwm_physical_support_gate_v1.yaml`;
- raw/aggregate runs: `runs/g2_ipwm_d3_physics_spectrum_audit_20260828/`;
- summary: `runs/g2_ipwm_d3_physics_spectrum_audit_20260828/physical_support_gate_summary.json`;
- summarizer: `scripts/summarize_ipwm_physical_support_gate.py`.
