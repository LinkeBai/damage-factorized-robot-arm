from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_31_subsets_receive_explicit_offline_path_gates(tmp_path: Path) -> None:
    output = tmp_path / "gates.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/audit_multilock_mujoco_gates.py"),
        "--reachability", str(ROOT / "results/real_robot/multilock_push_reachability_20260903.json"),
        "--model", str(ROOT / "sim/assets/genkiarm_push.xml"),
        "--safety", str(ROOT / "hardware/safety_limits.yaml"),
        "--samples", "11", "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 31
    assert all(row["joint_limit_gate"] in {"PASS", "FAIL"} for row in payload["rows"])
    assert all(row["linear_path_collision_gate"] in {"PASS", "FAIL"} for row in payload["rows"])
    assert len(payload["model_sha256"]) == 64
    assert payload["hardware_accessed"] is False
