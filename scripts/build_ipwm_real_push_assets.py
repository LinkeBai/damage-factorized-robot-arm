"""Build paper-facing assets for the audited Level-A IPWM real-Push panel."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


CORE = ("real-ipwm-intact-003", "real-ipwm-D2-003", "real-ipwm-D3-001-exec", "real-ipwm-D4-002")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.session / "ipwm_trials"
    rows = []
    for trial_id in CORE:
        trial = root / trial_id
        manifest = json.loads((trial / "model_decision_manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((trial / "model_decision_audit.json").read_text(encoding="utf-8"))
        telemetry = list(csv.DictReader((trial / "servo_telemetry.csv").open(encoding="utf-8")))
        joint = {"D2": "j2", "D3": "j3"}.get(manifest["condition"])
        drift = None
        if joint:
            drift = max(abs(int(r[f"{joint}_position_raw"]) - int(r[f"{joint}_target_raw"])) for r in telemetry)
        outcome = manifest["outcome"]
        rows.append({
            "trial_id": trial_id,
            "condition": manifest["condition"],
            "candidate_count": manifest["decision"]["candidate_count"],
            "audit_status": audit["status"],
            "success_axis5": outcome["success"],
            "endpoint_dx_px": outcome["endpoint_error_xy_px"][0],
            "endpoint_dy_px": outcome["endpoint_error_xy_px"][1],
            "endpoint_radial_px": outcome["endpoint_error_radial_px"],
            "displacement_px": outcome["displacement_px"],
            "locked_joint_max_drift_ticks": drift,
            "telemetry_samples": len(telemetry),
            "overhead_video": str((trial / "daheng_FDE23080341_raw.avi").resolve()),
            "wrist_video": str((trial / "directshow_index1_raw.avi").resolve()),
            "frame_timestamps": str((trial / "frame_timestamps.csv").resolve()),
            "commands": str((trial / "commands.csv").resolve()),
            "telemetry": str((trial / "servo_telemetry.csv").resolve()),
            "tracking": str((trial / "offline_cube_tracking_v1/yellow_cube_summary.json").resolve()),
        })
    if any(r["audit_status"] != "PASS" for r in rows):
        raise SystemExit("refusing to build paper assets: a core provenance audit is not PASS")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ipwm_real_push_core.json").write_text(json.dumps({
        "claim_level": "Level A feasibility only (core intact/D2/D3 plus pre-audited D4 extension)",
        "physical_n_per_condition": 1,
        "success_gate": "axis-wise abs(dx)<=5 px and abs(dy)<=5 px",
        "not_supported": ["statistical superiority", "SOTA superiority", "visual closed-loop control", "random-fault generalization"],
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "ipwm_real_push_core.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.axvspan(-5, 5, color="#d8f3dc", alpha=.7); ax.axhspan(-5, 5, color="#d8f3dc", alpha=.7)
    colors = {"intact": "#277da1", "D2": "#f8961e", "D3": "#9b5de5", "D4": "#e63946"}
    for row in rows:
        ax.scatter(row["endpoint_dx_px"], row["endpoint_dy_px"], s=90, color=colors[row["condition"]], label=row["condition"])
        ax.annotate(row["condition"], (row["endpoint_dx_px"], row["endpoint_dy_px"]), xytext=(5, 5), textcoords="offset points")
    ax.axvline(-5, color="#40916c", lw=1); ax.axvline(5, color="#40916c", lw=1)
    ax.axhline(-5, color="#40916c", lw=1); ax.axhline(5, color="#40916c", lw=1)
    ax.set(xlabel="Endpoint error dx (px)", ylabel="Endpoint error dy (px)", title="Audited real Push endpoints (n=1/condition)")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(args.output_dir / "ipwm_real_push_endpoint_errors.png", dpi=240); plt.close(fig)
    lines = ["# Audited IPWM real-Push video index", "", "Level-A feasibility only; one physical success per condition. Candidate count is a planning budget, not repeated trials.", ""]
    for row in rows:
        lines += [f"## {row['condition']} — `{row['trial_id']}`", "", f"- Result: PASS, error `({row['endpoint_dx_px']:.2f}, {row['endpoint_dy_px']:.2f}) px`", f"- Candidates: `{row['candidate_count']}`", f"- Overhead: `{row['overhead_video']}`", f"- Wrist: `{row['wrist_video']}`", f"- Commands/telemetry/tracking: `{row['commands']}`, `{row['telemetry']}`, `{row['tracking']}`", ""]
    (args.output_dir / "IPWM-REAL-PUSH-VIDEO-INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "conditions": len(rows), "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
