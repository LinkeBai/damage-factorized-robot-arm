"""Build Level-A physical-feasibility figure and LaTeX table from real data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    level = payload.get("physical_feasibility_claim_level")
    if level not in {"pilot", "formal"}:
        raise SystemExit("Cannot build feasibility assets without physical evidence")
    by_condition = payload.get("physical_feasibility_by_condition", {})
    conditions = [name for name in ("intact", "D2", "D3")
                  if by_condition.get(name, {}).get("valid_trials", 0) > 0]
    if not conditions:
        raise SystemExit("No condition-level physical feasibility rows")

    endpoint = np.asarray([
        by_condition[name]["mean_endpoint_error_m"] for name in conditions]) * 1000.0
    rates = {
        "Reach": np.asarray([by_condition[name]["reach_rate"] for name in conditions]) * 100,
        "Contact": np.asarray([by_condition[name]["contact_rate"] for name in conditions]) * 100,
        "Success": np.asarray([by_condition[name]["success_rate"] for name in conditions]) * 100,
    }
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), constrained_layout=True)
    x = np.arange(len(conditions))
    axes[0].bar(x, endpoint, color="#1f4e79", width=0.58)
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("Mean endpoint error (mm)")
    axes[0].set_title("Fixed-trajectory physical outcome")
    width = 0.23
    for offset, (label, values) in zip((-width, 0, width), rates.items()):
        axes[1].bar(x + offset, values, width, label=label)
    axes[1].set_xticks(x, conditions)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Reach-contact-success chain", pad=25)
    axes[1].legend(frameon=False, ncol=3, fontsize=7, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, -0.015,
             "Physical feasibility only - no learned-method comparison",
             ha="center", va="top", fontsize=7, color="#7a1f1f")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "% Generated from validated Level-A real-robot summary; do not edit manually.",
        "\\begin{tabular}{lrrrrrr}\\toprule",
        "Condition & Valid & Abort & Lock max (deg) & Endpoint (mm) & Contact (\\%) & Success (\\%)\\\\\\midrule",
    ]
    for name in conditions:
        item = by_condition[name]
        lock_deg = np.degrees(item["max_lock_error_rad"])
        lines.append(
            f"{name} & {item['valid_trials']} & {item['aborted_trials']} & "
            f"{lock_deg:.2f} & {1000 * item['mean_endpoint_error_m']:.2f} & "
            f"{100 * item['contact_rate']:.1f} & {100 * item['success_rate']:.1f}\\\\"
        )
    lines.extend([
        "\\bottomrule\\end{tabular}",
        f"% Physical feasibility claim level: {level}.",
        "% This table does not support learned-method superiority or sim-to-real control.",
    ])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(args.figure), "table": str(args.table),
                      "conditions": conditions, "claim_level": level}, indent=2))


if __name__ == "__main__":
    main()
