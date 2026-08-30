# External-reproduction patches

## TD-MPC2 original-arm adapter

- Upstream: <https://github.com/nicklashansen/tdmpc2>
- Audited base commit: `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`
- Upstream license: MIT, Copyright Nicklas Hansen (2023)
- Local patch: `tdmpc2-original-arm.patch`

The patch connects the separately downloaded TD-MPC2 checkout to this
repository's original 5-DoF environment and preserves the environment-provided
directional seed policy. The full upstream repository, checkpoints and replay
data remain local and are not vendored here.

Apply from the root of the audited TD-MPC2 checkout:

```text
git apply --ignore-whitespace /path/to/damage-factorized-robot-arm/third_party/patches/tdmpc2-original-arm.patch
```
