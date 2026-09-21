"""Create two verified, non-overlapping copies of the final Push-v2 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "data/real_robot/final_push_v2_trials",
    "data/real_robot/session_20260901/ipwm_preparations",
    "reports/ipwm_final_submission_archive_ready_20260905",
    "reports/ipwm_final_submission_sealed_20260905_v3",
    "src", "scripts", "config", "hardware",
)
SOURCE_FILES = (
    "data/real_robot/final_push_formal_membership_v4.csv",
    "data/real_robot/final_push_formal_membership_v4_provenance.json",
    "results/real_robot/final_shutdown_20260905/summary.json",
    "runs/icra_confirmation_d3_query_selective_w10/seed27/model.pt",
    "results/real_robot/push_axis_current_epoch_20260903.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup-a", type=Path, required=True)
    ap.add_argument("--backup-b", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    repo = ROOT.resolve()
    destinations = [args.backup_a.resolve(), args.backup_b.resolve()]
    roots = [repo, *destinations]
    for i, left in enumerate(roots):
        for right in roots[i + 1:]:
            try:
                left.relative_to(right); overlap = True
            except ValueError:
                try:
                    right.relative_to(left); overlap = True
                except ValueError:
                    overlap = False
            if overlap:
                raise SystemExit(f"backup roots overlap: {left} and {right}")
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an existing backup destination")
    files: set[Path] = set()
    for relative in SOURCE_ROOTS:
        source = repo / relative
        if not source.is_dir():
            raise SystemExit(f"missing source directory: {source}")
        files.update(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for relative in SOURCE_FILES:
        source = repo / relative
        if not source.is_file():
            raise SystemExit(f"missing source file: {source}")
        files.add(source)
    ordered = sorted(files)
    entries = []
    for index, source in enumerate(ordered, 1):
        entries.append({"relative_path": source.relative_to(repo).as_posix(),
                        "bytes": source.stat().st_size, "sha256": sha256(source)})
        if index % 250 == 0:
            print(f"source_hash_progress={index}/{len(ordered)}", flush=True)
    for destination in destinations:
        destination.mkdir(parents=True)
        for index, entry in enumerate(entries, 1):
            source = repo / entry["relative_path"]
            target = destination / entry["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                raise SystemExit(f"copy verification failed: {target}")
            if index % 250 == 0:
                print(f"copy_verify_progress={destination.name}:{index}/{len(entries)}", flush=True)
        with (destination / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("relative_path", "bytes", "sha256"))
            writer.writeheader(); writer.writerows(entries)
    manifest_hashes = [sha256(path / "SHA256SUMS.csv") for path in destinations]
    source_unchanged = all((repo / e["relative_path"]).stat().st_size == e["bytes"]
                           and sha256(repo / e["relative_path"]) == e["sha256"] for e in entries)
    receipt = {
        "status": "TWO_COPIES_VERIFIED_AGAINST_FROZEN_SOURCE" if source_unchanged and len(set(manifest_hashes)) == 1 else "FAIL_CLOSED",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "backup_a": str(destinations[0]), "backup_b": str(destinations[1]),
        "source_file_count": len(entries), "source_bytes": sum(e["bytes"] for e in entries),
        "manifest_sha256": manifest_hashes[0], "source_unchanged_after_copy": source_unchanged,
        "physical_shutdown_verified": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for destination in destinations:
        (destination / "ARCHIVE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt["status"].startswith("TWO_COPIES") else 2


if __name__ == "__main__":
    raise SystemExit(main())
