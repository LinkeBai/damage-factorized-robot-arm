from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_single_lock_safety.py"


def test_dry_run_never_opens_serial_and_records_plan(tmp_path: Path) -> None:
    out = tmp_path / "probe"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--joint", "j1", "--out", str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"status": "DRY_RUN"' in text
    assert '"all_arm_joints_powered": false' in text
    assert not (out / "telemetry.csv").exists()


def test_rejects_excessive_probe_amplitude(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--joint", "j5", "--amplitude-deg", "4",
         "--out", str(tmp_path / "probe")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "amplitude must be" in result.stderr
