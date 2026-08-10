"""Build reproducibility manifests for completed G1 runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run_roots = [ROOT / "runs" / "g1_split", ROOT / "runs" / "g1_control_gate"]
    records = []
    for root in run_roots:
        for summary_path in sorted(root.glob("*/summary.json")):
            run_dir = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = {
                "stage": "G1",
                "run_dir": str(run_dir.relative_to(ROOT)),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": "recorded_at_closeout_pending_worktree_audit",
                "seeds": summary.get("seeds", []),
                "protocol": summary.get("protocol", {}),
                "target_split": summary.get("target_split", {}),
                "settings": summary.get("settings", {}),
                "compute": summary.get("compute", {}),
                "claim_scope": summary.get("claim_scope", "G1 simulation evidence"),
                "source_summary": summary_path.name,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            records.append(manifest)
    out = ROOT / "results" / "final" / "g1-run-manifest-index.json"
    out.write_text(json.dumps({"stage": "G1", "runs": records}, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} manifests to {out}")


if __name__ == "__main__":
    main()
