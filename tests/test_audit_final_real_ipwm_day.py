import csv
import json
from pathlib import Path

import yaml

from scripts.audit_final_real_ipwm_day import REQUIRED_PUSH, audit


def write_fixture(root: Path):
    protocol = root / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"protocol_id": "p", "status": "frozen"}))
    schedule = root / "schedule.csv"
    fields = ["trial_id", "task", "distance_px", "condition", "repeat",
              "priority", "evidence_level", "status"]
    with schedule.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow(dict(zip(fields, ["P1", "push", "15", "D2", "1",
                                          "core", "quantitative_closed_loop", "GO"])))
        writer.writerow(dict(zip(fields, ["G1", "grasp", "", "D2", "1",
                                          "core", "feasibility", "NO_GO"])))
    evidence = root / "evidence"
    push = evidence / "P1"; push.mkdir(parents=True)
    for name in REQUIRED_PUSH:
        if name == "ipwm_closed_loop_evidence.json":
            (push / name).write_text(json.dumps({"status": "PASS",
                "is_genuine_physical_ipwm_closed_loop_evidence": True}))
        else:
            (push / name).write_bytes(b"x")
    (push / "ipwm_replan_cycles").mkdir()
    grasp = evidence / "G1"; grasp.mkdir()
    for name in ("run_manifest.json", "daheng_FDE23080341_raw.avi",
                 "directshow_index1_raw.avi", "servo_telemetry.csv"):
        (grasp / name).write_bytes(b"x")
    return protocol, schedule, evidence


def test_complete_packets_only_allow_archive_not_dismantle(tmp_path):
    protocol, schedule, evidence = write_fixture(tmp_path)
    result = audit(protocol, schedule, evidence)
    assert result["core_required_complete"] is True
    assert result["evidence_ready_for_archive"] is True
    assert result["ready_to_dismantle"] is False


def test_missing_closed_loop_evidence_blocks_completion(tmp_path):
    protocol, schedule, evidence = write_fixture(tmp_path)
    (evidence / "P1" / "ipwm_replan_cycles").rmdir()
    result = audit(protocol, schedule, evidence)
    assert result["ready_to_dismantle"] is False
    assert "missing:ipwm_replan_cycles" in result["details"][0]["errors"]


def test_excluded_condition_requires_retained_evidence(tmp_path):
    protocol, schedule, evidence = write_fixture(tmp_path)
    with schedule.open("a", newline="") as handle:
        csv.writer(handle).writerow(["X1", "push", "15", "J1+J2", "1",
                                    "conditional", "coverage", "PREFLIGHT_EXCLUDED"])
    result = audit(protocol, schedule, evidence)
    assert not result["evidence_ready_for_archive"]
    assert "missing:adjudication.json" in result["details"][-1]["errors"]
