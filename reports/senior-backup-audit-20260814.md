# Senior experiment backup audit (2026-08-14)

## Provenance

- Archive: `damage-factorized-robot-arm_experiment_backup_20260814.tar.gz`
- SHA-256: `B5168D34BCE7BCB95559DA499C9AEF2CB078C9648BEEDF50EB3F6B61C03CED41`
- Imported scope: source code, configuration, paper draft, tests, and `results/final/`
- Excluded scope: local `runs/` process artifacts

## Verification

- Full test suite: 114 passed
- MuJoCo Push environment: 14-dimensional state, 100-step smoke test passed
- Reach held-out result: 5 seeds are present and the supplied significance script reproduces a
  non-significant DFWM vs topology-only difference at K=5

## Evidence boundary

The archive records a preliminary Push multi-step improvement of 15.8% in
`EXPERIMENT-LOG.md` and supplies the Push benchmark implementation. It does not contain a Push
per-seed CSV or a Push run directory from which that number can be independently recomputed.
Treat 15.8% as a provisional observation until the six-method, five-seed Push benchmark is rerun
and its paired bootstrap confidence interval is committed under `results/final/`.
