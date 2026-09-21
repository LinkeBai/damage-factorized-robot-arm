"""Resumable execution loop for the frozen historical-positive replication."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ipwm_positive_scale_20260911"
PYTHON = ROOT / ".venv-cuda/Scripts/python.exe"
PHYSICS = ["nominal", "weak_motor", "high_damping", "delay_1", "noisy_deadband",
           "mixed_composition", "mixed_unseen"]
SEEDS = [27, 37, 47]


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs():
    for physics in PHYSICS:
        for shard in range(12):
            data = OUT / "data" / physics / f"shard-{shard:02d}.pt"
            yield (f"collect-{physics}-{shard:02d}",
                   [str(PYTHON), str(ROOT / "scripts/run_ipwm_positive_scale.py"), "--collect",
                    "--physics", physics, "--shard", str(shard)], [data, data.with_suffix(".json")])
    for seed in SEEDS:
        for physics in PHYSICS:
            for shard in range(12):
                result = OUT / "evaluation" / f"seed{seed}" / physics / f"shard-{shard:02d}.json"
                yield (f"evaluate-s{seed}-{physics}-{shard:02d}",
                       [str(PYTHON), str(ROOT / "scripts/run_ipwm_positive_scale.py"), "--evaluate",
                        "--seed", str(seed), "--physics", physics, "--shard", str(shard)], [result])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "loop-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    all_jobs = list(jobs())
    for index, (job_id, command, outputs) in enumerate(all_jobs):
        old = state["jobs"].get(job_id, {})
        if old.get("status") == "complete" and all(p.is_file() and sha(p) == old["hashes"][str(p.relative_to(ROOT))]
                                                     for p in outputs):
            continue
        state["jobs"][job_id] = {"status": "running", "started": time.time()}; write(state_path, state)
        log_path = OUT / "logs" / f"{job_id}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        complete = result.returncode == 0 and all(p.is_file() for p in outputs)
        state["jobs"][job_id] = {"status": "complete" if complete else "failed",
                                  "returncode": result.returncode, "ended": time.time(),
                                  "hashes": {str(p.relative_to(ROOT)): sha(p) for p in outputs if p.is_file()}}
        write(state_path, state)
        write(OUT / "loop-status.json", {"status": "running" if complete else "failed",
              "completed_jobs": sum(v["status"] == "complete" for v in state["jobs"].values()),
              "total_jobs": len(all_jobs), "last_job": job_id, "index": index})
        if not complete:
            raise RuntimeError(f"Job failed: {job_id}; inspect {log_path}")
    validation_log = OUT / "logs" / "final-validation.log"
    with validation_log.open("w", encoding="utf-8") as log:
        result = subprocess.run([str(PYTHON), str(ROOT / "scripts/validate_ipwm_positive_scale_completion.py")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    audit_path = OUT / "validated-completion.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    passed = result.returncode == 0 and audit.get("passed") is True
    write(OUT / "loop-status.json", {"status": "complete" if passed else "validation_failed",
          "goal_stage_complete": passed, "completed_jobs": len(all_jobs), "total_jobs": len(all_jobs),
          "validation_returncode": result.returncode})
    if not passed:
        raise RuntimeError(f"Final positive-scale validation failed; inspect {validation_log}")


if __name__ == "__main__":
    main()
