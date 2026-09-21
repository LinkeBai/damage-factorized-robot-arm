"""Strict completion gate for the frozen positive-scale replication."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ipwm_positive_scale_20260911"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    subprocess.run([sys.executable, "scripts/summarize_ipwm_positive_scale.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/render_ipwm_positive_scale.py"], cwd=ROOT, check=True)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_hashes"].items():
        assert sha(ROOT / name) == expected, f"Frozen source changed: {name}"
    summary = json.loads((OUT / "results-summary.json").read_text(encoding="utf-8"))
    checks = {
        "independent_test_count_exceeds_activepusher_4000": summary["independent_trajectories"] == 8400,
        "each_condition_exceeds_activepusher_per_condition_1000": summary["independent_trajectories_per_condition"] == 1200,
        "all_63_registered_cells_retained": summary["all_cells"] == 63 and summary["primary_cells"] == 27,
        "all_8400_trajectories_unique": summary["dataset_unique"] and summary["overlap_with_previous_700"] == 0,
        "selective_state_isolation_verified": summary["max_robot_change_from_carrier"] <= 1e-8,
        "locked_constraint_verified": summary["max_lock_violation"] <= 1e-8,
    }
    records = {}
    for physics in protocol["physics"]:
        for shard in range(protocol["shards_per_condition"]):
            for suffix in ["pt", "json"]:
                path = OUT / "data" / physics / f"shard-{shard:02d}.{suffix}"
                assert path.is_file(); records[str(path.relative_to(ROOT))] = sha(path)
            for seed in protocol["seeds"]:
                path = OUT / "evaluation" / f"seed{seed}" / physics / f"shard-{shard:02d}.json"
                assert path.is_file(); records[str(path.relative_to(ROOT))] = sha(path)
    artifacts = [
        OUT / "all-cells.csv", OUT / "results-summary.json", OUT / "positive-replication.png",
        OUT / "positive-replication.svg", ROOT / "paper/ipwm-positive-scale-evidence-20260911.md",
        ROOT / "paper/ipwm-positive-scale-evidence-20260911.html",
        ROOT / "paper/ipwm-positive-scale-experiments-20260911.tex",
    ]
    for path in artifacts:
        assert path.is_file() and path.stat().st_size > 100, str(path)
        records[str(path.relative_to(ROOT))] = sha(path)
    manifest = {"files": records, "protocol_sha256": sha(OUT / "protocol.json"), "checks": checks,
                "counting_scope": protocol["counting"], "claim_scope": protocol["scope"]}
    (OUT / "reproducibility-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {"passed": all(checks.values()), "checks": checks,
              "manifest_sha256": sha(OUT / "reproducibility-manifest.json")}
    (OUT / "validated-completion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("Positive-scale completion conditions were not met")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
