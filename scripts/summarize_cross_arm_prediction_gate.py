"""Aggregate the frozen three-seed cross-arm robot-transition Gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    decisions = []
    improvements = []
    for row in rows:
        structured = row["shared_structured"]
        flat = row["flat_unstructured"]
        improvement = float(row["structured_relative_pooled_improvement"])
        both = all(structured[robot] < flat[robot] for robot in ("genkiarm", "panda"))
        passed = improvement >= 0.10 and both
        improvements.append(improvement)
        decisions.append({
            "seed": row["seed"], "pooled_relative_improvement": improvement,
            "genkiarm_improved": structured["genkiarm"] < flat["genkiarm"],
            "panda_improved": structured["panda"] < flat["panda"],
            "seed_passed": passed,
        })
    output = {
        "version": "cross_arm_prediction_gate_summary_v1",
        "scope": "joint_transition_only_joint_training_on_both_robot_structures",
        "decisions": decisions,
        "positive_seeds": sum(row["seed_passed"] for row in decisions),
        "total_seeds": len(decisions),
        "mean_pooled_relative_improvement": float(np.mean(improvements)),
        "gate_passed": sum(row["seed_passed"] for row in decisions) >= 2,
        "promotion": "object_contact_gate_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
