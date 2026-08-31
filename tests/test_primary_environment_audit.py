import json
from pathlib import Path
import subprocess
import sys


def test_active_primary_environment_matches_frozen_lock(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "environment.json"
    completed = subprocess.run(
        [sys.executable, "scripts/audit_primary_environment.py", "--output", str(output)],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert all(item["match"] for item in payload["packages"].values())
    assert payload["accelerator"]["cuda_available"] is True
