"""Resumable, frozen GenkiArm five-seed confirmation pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
XML = Path("sim/assets/genkiarm_push.xml")
V0 = Path("runs/g2_dual_expert_fair_gate_v0/seed7_v1")
SEEDS = (107, 117, 127, 137, 147)
STAGES = ("base", "zero", "context", "adapter", "contact", "physical", "evaluate")


def _run_dir(stage: str, seed: int) -> Path:
    roots = {
        "base": "runs/g2_ipwm_genkiarm_base_v2",
        "zero": "runs/g2_ipwm_genkiarm_zero_topology_v2",
        "context": "runs/g2_ipwm_genkiarm_context_v2",
        "adapter": "runs/g2_ipwm_genkiarm_adapter_v2",
        "contact": "runs/g2_ipwm_genkiarm_contact_residual_v2",
        "physical": "runs/g2_ipwm_genkiarm_physical_context_v2",
        "evaluate": "runs/g2_ipwm_genkiarm_confirmation_v2",
    }
    return Path(roots[stage]) / f"seed{seed}_v1"


def _complete(stage: str, seed: int) -> bool:
    directory = _run_dir(stage, seed)
    summary = directory / ("metrics.json" if stage == "evaluate" else "summary.json")
    required = {
        "base": ("model.pt", "baseline_model.pt"), "zero": ("model.pt",),
        "context": ("context_encoder.pt",), "adapter": ("shared_adapter.pt", "bt_adapter.pt"),
        "contact": ("model.pt",), "physical": ("model.pt",),
        "evaluate": ("raw.json",),
    }[stage]
    if not summary.is_file() or not all((directory / name).is_file() for name in required):
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (payload.get("seed") == seed and not payload.get("smoke", False)
            and str(payload.get("xml", "")).replace("\\", "/").endswith(str(XML).replace("\\", "/")))


def _command(stage: str, seed: int) -> list[str]:
    py = sys.executable; cache = "runs/cache/genkiarm_confirmation_v2"
    if stage in {"base", "zero", "contact", "physical"}:
        configs = {
            "base": "g2_ipwm_genkiarm_base_train_v2.yaml",
            "zero": "g2_ipwm_genkiarm_zero_topology_v2.yaml",
            "contact": "g2_ipwm_genkiarm_contact_residual_v2.yaml",
            "physical": "g2_ipwm_genkiarm_physical_context_v2.yaml",
        }
        return [py, "scripts/run_bt_dpwm_gate_y0.py", "--config", f"config/experiment/{configs[stage]}",
                "--seed", str(seed), "--v0-run-dir", str(V0), "--xml", str(XML),
                "--cache-dir", cache, "--output-dir", str(_run_dir(stage, seed))]
    if stage == "context":
        return [py, "scripts/train_physical_context_encoder_z64.py", "--config",
                "config/experiment/g2_ipwm_genkiarm_context_encoder_v2.yaml", "--seed", str(seed),
                "--xml", str(XML), "--output-dir", str(_run_dir(stage, seed))]
    if stage == "adapter":
        return [py, "scripts/run_bt_dpwm_fewshot_z48.py", "--config",
                "config/experiment/g2_bt_dpwm_known_topology_z63_v1.yaml", "--seed", str(seed),
                "--xml", str(XML), "--base-run", str(_run_dir("base", seed)),
                "--bt-run", str(_run_dir("zero", seed)), "--train-only",
                "--output-dir", str(_run_dir(stage, seed))]
    directory = _run_dir(stage, seed)
    return [py, "scripts/evaluate_ipwm_selective_rollout.py", "--config",
            "config/experiment/g2_ipwm_genkiarm_evaluation_v2.yaml", "--seed", str(seed),
            "--model", str(_run_dir("physical", seed) / "model.pt"),
            "--output", str(directory / "metrics.json"), "--raw-output", str(directory / "raw.json"),
            "--cache-dir", cache, "--query-seed-base", str(seed + 1000), "--xml", str(XML)]


def _write_execution_manifest(seed: int) -> None:
    root = Path("runs/g2_ipwm_genkiarm_confirmation_v2") / f"seed{seed}_v1"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "genkiarm_confirmation_v2_execution_manifest",
        "seed": seed, "xml": str(XML), "query_seed_base": seed + 1000,
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stages": list(STAGES),
    }
    (root / "execution_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _execute(command: list[str], log_path: Path) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        code = process.wait()
    finished_at = datetime.now(timezone.utc)
    timing = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "exit_code": code,
    }
    if code:
        raise subprocess.CalledProcessError(code, command)
    return timing


def _record_stage_timing(stage: str, seed: int, command: list[str],
                         timing: dict[str, object]) -> None:
    """Persist runtime telemetry without changing the frozen experiment recipe."""
    path = _run_dir(stage, seed) / "stage_timing.json"
    payload = {"stage": stage, "seed": seed, "command": command, **timing}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--start-stage", choices=STAGES, default="base")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); seeds = (args.seed,) if args.seed else SEEDS
    start = STAGES.index(args.start_stage)
    for seed in seeds:
        _write_execution_manifest(seed)
        for stage in STAGES[start:]:
            if _complete(stage, seed):
                print(f"[resume] seed={seed} stage={stage} complete", flush=True); continue
            command = _command(stage, seed)
            print("[run] " + subprocess.list2cmdline(command), flush=True)
            if args.dry_run: continue
            timing = _execute(command, _run_dir(stage, seed) / "stage.log")
            _record_stage_timing(stage, seed, command, timing)
            print(f"[timing] seed={seed} stage={stage} "
                  f"duration={timing['duration_seconds']:.1f}s", flush=True)
            if not _complete(stage, seed):
                raise RuntimeError(f"stage finished without a valid manifest: seed={seed} stage={stage}")


if __name__ == "__main__": main()
