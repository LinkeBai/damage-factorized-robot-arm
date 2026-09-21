# Configuration and model source snapshot

This local archive contains the exact nominal MuJoCo XML, lock construction, model source, matched-adaptation configuration, data generation and evaluation scripts read from the current workspace. File hashes are in manifest.json.

The configuration-source archive alone is a configuration disclosure snapshot, not a standalone runnable release. Separate archives below provide weights and calibration evaluation records. Full training datasets, all transitive imports, environment dependencies and a verified pretraining recipe remain outside the verified release scope. Current sources must be checked against each historical experiment manifest before treating them as the executed version.

The scale experiment shares an already trained seed-27 primary model across adaptation fits, including trained robot and object dynamics. Historical coupled models and hardware deployment are separate model families. See the manuscript appendix and ../evidence-source-index.md for the corresponding evidence and limitations.

This archive has not been uploaded or submitted to a venue, and does not establish anonymity clearance.

## Foundation checkpoint archive

foundation-checkpoints.zip supplies the three seed-27 files loaded by the training helper. Archive contents match the current files, and the full source checkpoint matches the frozen training protocol hash. foundation-checkpoints.json records their identities. Later adapted weights, data, transitive dependencies and independent execution verification remain outside this archive.

## Adapted models

adapted-checkpoints.zip contains 18 matched adaptation checkpoints, six calibrated checkpoints and six validation selections. Each selection source hash matches its original IPWM checkpoint. File hashes are in adapted-checkpoints.json. Selection records preserve original local paths and have not been anonymized. No upload has occurred.

## Complete calibration evaluation records

calibration-evaluation-records.zip includes all 240 trajectory cells, sidecars, frozen protocol, full paired report, selection records and analysis/audit sources. It supports reanalysis of this closed-loop confirmation, not reconstruction of the entire training dataset. Paths preserve the project layout. Some source paths in metadata are local; the package has not been anonymized or uploaded.

## Verified calibration reanalysis

Extract calibration-evaluation-records.zip into an empty directory, preserving its paths. From that directory run:

```text
python work/experiment_revision/analyze_scale_planning.py
```

The script checks all 240 cells and frozen script/selection identities before reporting effects. It writes runs/ipwm_validation_scale_calibration/planning_confirmation/paired-analysis.json. The isolated-directory check reproduced the archived report byte for byte. See reanalysis-check.json for the verified SHA-256.

This command uses saved trajectories and NumPy; it does not execute MuJoCo or train a model. The separate integrity script additionally needs checkpoint paths from the weight archives. Training and simulation dependencies have not been validated as a standalone environment.

## Verified checkpoint and trajectory integrity

Extract adapted-checkpoints.zip into the same directory as calibration-evaluation-records.zip, then run `python work/experiment_revision/audit_scale_planning_integrity.py`. An isolated-directory check passed all 240 cells, including actual checkpoint hashes, locked commands, endpoint/success recomputation and shared initial conditions. See isolated-integrity-check.json. This verifies saved artifacts; it does not rerun training or simulation.

## Carrier specification and prior training

[carrier-guide.md](carrier-guide.md) specifies each network component, active constructor settings, variant initialization and checkpoint route. [carrier-specification.json](carrier-specification.json) records strict saved-tensor compatibility, complete effective constructor arguments, parameter shapes/counts and source hashes. This is an inspection of stored tensors, not a new model evaluation.

[source-training-provenance.md](source-training-provenance.md) distinguishes the archived training recipe, retained data files and exact tensor comparisons from information the archive cannot establish. It covers training/development before the additional-data experiments. In particular, neither the name `foundation-checkpoints.zip` nor a later 1,000-trajectory budget means the source model was trained from scratch on that amount of data.

These new documentation files sit beside the existing archives and are not yet embedded in those ZIP files. Existing ZIP contents and their recorded hashes remain unchanged. No public release or independent from-scratch reproduction is implied.
