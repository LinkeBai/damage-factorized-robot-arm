"""Build real-robot result figure and LaTeX table from a validated summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ci_error(mean: float, ci: list[float] | None) -> tuple[float, float]:
    if ci is None or any(not np.isfinite(value) for value in ci):
        return 0.0, 0.0
    return max(0.0, mean - ci[0]), max(0.0, ci[1] - mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if payload.get("claim_level") == "no paired evidence":
        raise SystemExit("Cannot build paper assets without paired evidence")
    by_condition = payload.get("paired_by_condition", {})
    conditions = [name for name in ("intact", "D2", "D3")
                  if by_condition.get(name, {}).get("pairs", 0) > 0]
    if not conditions:
        raise SystemExit("No condition-level paired summaries available")

    endpoint = [by_condition[name]["endpoint_improvement_m"]["mean"] for name in conditions]
    endpoint_err = np.asarray([
        ci_error(value, by_condition[name]["endpoint_improvement_m"]["ci95"])
        for name, value in zip(conditions, endpoint)
    ]).T
    success = [by_condition[name]["success_improvement"]["mean"] for name in conditions]
    reach = [by_condition[name]["reach_improvement"]["mean"] for name in conditions]
    contact = [by_condition[name]["contact_improvement"]["mean"] for name in conditions]

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), constrained_layout=True)
    x = np.arange(len(conditions))
    axes[0].errorbar(x, np.asarray(endpoint) * 1000.0,
                     yerr=endpoint_err * 1000.0, fmt="o", color="#1f4e79",
                     capsize=3, linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("Paired endpoint improvement (mm)")
    axes[0].set_title("Continuous outcome (mean, 95% bootstrap CI)")
    width = 0.24
    axes[1].bar(x - width, np.asarray(reach) * 100, width, label="Reach")
    axes[1].bar(x, np.asarray(contact) * 100, width, label="Contact")
    axes[1].bar(x + width, np.asarray(success) * 100, width, label="Success")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, conditions)
    axes[1].set_ylabel("Candidate-reference change (pp)")
    axes[1].set_title("Six-stage physical outcomes")
    axes[1].legend(frameon=False, ncol=3, fontsize=7, loc="best")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)

    reference = payload["paired_comparison"]["reference_method"]
    candidate = payload["paired_comparison"]["candidate_method"]
    lines = [
        "% Generated from validated real-robot summary; do not edit numbers manually.",
        "\\begin{tabular}{lrrrrr}\\toprule",
        "Condition & Pairs & Endpoint (mm) & Reach (pp) & Contact (pp) & Success (pp)\\\\\\midrule",
    ]
    for name in conditions:
        item = by_condition[name]
        lines.append(
            f"{name} & {item['pairs']} & "
            f"{1000 * item['endpoint_improvement_m']['mean']:.2f} & "
            f"{100 * item['reach_improvement']['mean']:.1f} & "
            f"{100 * item['contact_improvement']['mean']:.1f} & "
            f"{100 * item['success_improvement']['mean']:.1f}\\\\"
        )
    lines.extend([
        "\\bottomrule\\end{tabular}",
        f"% Positive values favor {candidate} over {reference}.",
        f"% Claim level: {payload['claim_level']}; raw-file check: "
        f"{payload['all_required_files_checked']}.",
    ])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "figure": str(args.figure), "table": str(args.table),
        "conditions": conditions, "claim_level": payload["claim_level"],
    }, indent=2))


if __name__ == "__main__":
    main()
