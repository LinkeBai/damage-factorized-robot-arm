import json
from pathlib import Path
import subprocess
import sys


def test_current_icra_pdf_passes_anonymity_and_format_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [sys.executable, "scripts/audit_icra_pdf.py", "paper/main.pdf",
         "--source", "paper/main.tex", "--output", str(output)],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["pages"] <= 8
    assert payload["forbidden_identity_matches"] == []
    assert all(font["embedded"] == "yes" for font in payload["fonts"])
