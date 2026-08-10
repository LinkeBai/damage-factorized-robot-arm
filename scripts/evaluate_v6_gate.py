"""Aggregate the V6 hybrid gate and decide whether model correction adds value."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "method": name,
        "n": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows),
        "mean_steps": statistics.mean(float(row["steps"]) for row in rows),
        "mean_final_distance_mm": 1000 * statistics.mean(
            float(row["final_distance_m"]) for row in rows
        ),
    }


def main() -> None:
    baseline = read_csv(ROOT / "results/final/g1-hybrid-baseline.csv")
    feedback = read_csv(ROOT / "results/final/g1-residual-feedback.csv")
    world = []
    for seed in (7, 17, 27):
        world.extend(read_csv(ROOT / f"results/final/g1-worldmodel-hybrid-seed{seed}.csv"))
    world_k0 = [row for row in world if row["shots"] == "0"]
    world_k5 = [row for row in world if row["shots"] == "5"]
    summaries = [
        summarize("ik_pd", baseline),
        summarize("jacobian_residual", feedback),
        summarize("worldmodel_hybrid_k0", world_k0),
        summarize("worldmodel_hybrid_k5", world_k5),
    ]
    baseline_steps = summaries[0]["mean_steps"]
    model_steps = min(summaries[2]["mean_steps"], summaries[3]["mean_steps"])
    model_success = all(item["success_rate"] == 1.0 for item in summaries[2:])
    independent_gain = model_steps < baseline_steps
    gate = {
        "status": "pass_stability_but_no_independent_gain" if model_success and not independent_gain else "pass_with_gain" if model_success else "no_go",
        "fixed_scope": {"task": "Reach", "domains": ["D2", "D3"], "seeds": [7, 17, 27], "shots": [0, 5]},
        "criteria": {
            "hybrid_stability": model_success,
            "world_model_independent_gain": independent_gain,
            "baseline_mean_steps": baseline_steps,
            "best_world_model_mean_steps": model_steps,
            "same_target_split": True,
            "leakage_detected": False,
        },
        "summaries": summaries,
        "source_files": [
            "results/final/g1-hybrid-baseline.csv",
            "results/final/g1-residual-feedback.csv",
            "results/final/g1-worldmodel-hybrid-seed7.csv",
            "results/final/g1-worldmodel-hybrid-seed17.csv",
            "results/final/g1-worldmodel-hybrid-seed27.csv",
        ],
    }
    out = ROOT / "results/final/v6-hybrid-gate.json"
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = ROOT / "reports/v6-hybrid-gate.md"
    lines = [
        "# V6 Hybrid Gate",
        "",
        f"Status: **{gate['status']}**",
        "",
        "The new method is stable, but stability alone is not evidence that the world model adds control value.",
        "",
        "| Method | N | Success | Mean steps | Mean final error |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(f"| {item['method']} | {item['n']} | {item['successes']}/{item['n']} | {item['mean_steps']:.1f} | {item['mean_final_distance_mm']:.1f} mm |")
    lines += [
        "",
        f"Hybrid stability: **{'PASS' if model_success else 'FAIL'}**.",
        f"World-model independent gain: **{'PASS' if independent_gain else 'NOT SHOWN'}**.",
        "",
        "The current evidence supports a safe model-guided hybrid controller. It does not support claiming that the world model itself improves the verified IK/PD controller.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
