"""Compare the active runtime with the frozen primary-evidence environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path,
        default=Path("config/environment/primary-environment-lock.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/final/primary-environment-audit.json"),
    )
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    errors: list[str] = []
    actual_python = platform.python_version()
    if actual_python != lock["python"]:
        errors.append(f"python: expected {lock['python']}, found {actual_python}")
    packages = {}
    for name, expected in lock["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    accelerator = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected_accelerator = lock["accelerator"]
    if str(accelerator["torch_cuda"]) != expected_accelerator["cuda_runtime_reported_by_torch"]:
        errors.append("torch CUDA runtime does not match lock")
    if accelerator["cudnn"] != expected_accelerator["cudnn"]:
        errors.append("cuDNN version does not match lock")
    commands = {}
    for name in lock["external_commands"]:
        path = shutil.which(name)
        commands[name] = {
            "available": path is not None,
            "resolved_executable": Path(path).name if path is not None else None,
        }
        if path is None:
            errors.append(f"external command missing: {name}")
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "lock": str(args.lock),
        "python": {"expected": lock["python"], "actual": actual_python},
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
        "external_commands": commands,
        "python_executable_name": Path(sys.executable).name,
        "errors": errors,
        "scope_note": (
            "Exact lock documents the evidence-producing host. CPU/CUDA metric "
            "equivalence was separately audited; the named GPU is not required "
            "for numerical correctness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
