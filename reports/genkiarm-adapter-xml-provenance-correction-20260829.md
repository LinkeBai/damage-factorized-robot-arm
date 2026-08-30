# GenkiArm adapter XML provenance correction (2026-08-29)

## Finding

During the first formal seed-107 adapter run, runtime profiling exposed that
`run_bt_dpwm_fewshot_z48.py` passed `genkiarm_push.xml` to the Warp active-probe
collector but omitted the XML argument from the CPU goal-query and validation
collectors. Those calls therefore fell back to `arm_push.xml`. A mixed-robot
adapter would not be valid calibrated-GenkiArm evidence.

## Containment

- The run was interrupted before any adapter checkpoint or summary was
  produced.
- The one completed legacy-format query cache was moved, without deletion, to
  `runs/quarantine/invalid_simplified_xml_adapter_cache_20260829/` and is
  excluded from all evidence.
- Completed seed-107 base, zero-topology and context artifacts are unaffected;
  each of those stages used the explicit calibrated XML.

## Correction

- Every CPU goal-query, validation, calibration and test collection now
  receives the resolved explicit XML.
- Query-cache filenames contain a hash of the resolved XML path and file
  contents.
- Legacy parent caches, which lack XML provenance, can be reused only for the
  original `arm_push.xml`, never for a calibrated robot asset.
- `tests/test_fewshot_goal_query_xml_provenance.py` statically verifies that
  every CPU collector call has an explicit `xml_path`, that both cache types
  are XML-namespaced, and that legacy reuse is guarded.

Eight relevant provenance/pipeline/model-contract tests passed before the
clean adapter restart. The restarted seed-107 adapter run must produce a new
XML-namespaced cache and a summary naming `sim/assets/genkiarm_push.xml` before
it is admissible.

## Claim boundary

The interrupted mixed-XML collection is an implementation No-Go and contributes
no paper result. It is retained in the audit trail because silently accepting
it would invalidate the actual-GenkiArm evidence claim.
