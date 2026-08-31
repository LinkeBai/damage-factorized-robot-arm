import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def run_schedule(root: Path, output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_real_robot_push_schedule.py",
         "--seed", "20260901", "--fault-pairs", "2", "--intact-pairs", "1",
         "--output", str(output)],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))


def test_schedule_is_deterministic_paired_and_hashed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first_meta, second_meta = run_schedule(root, first), run_schedule(root, second)
    assert first.read_bytes() == second.read_bytes()
    expected_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_meta["sha256"] == second_meta["sha256"] == expected_hash
    with first.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    grouped = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["pair_id"]), []).append(row)
    assert len(grouped) == 5
    for pair in grouped.values():
        assert {row["method"] for row in pair} == {"nominal", "global_matched"}
        assert len({row["position_id"] for row in pair}) == 1
    assert [int(row["trial_order"]) for row in rows] == list(range(1, 11))


def test_schedule_rejects_single_method(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/generate_real_robot_push_schedule.py",
         "--methods", "nominal", "--output", str(tmp_path / "bad.csv")],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode != 0
