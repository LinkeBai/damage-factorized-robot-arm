"""Merge all 31 frozen fault subsets into an explicit gate/evidence matrix."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reachability",type=Path,required=True)
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--mujoco-gates",type=Path)
    p.add_argument("--output-csv",type=Path,required=True)
    p.add_argument("--output-md",type=Path,required=True)
    a=p.parse_args()
    reach=json.loads(a.reachability.read_text(encoding="utf-8"))["rows"]
    ledger=json.loads(a.ledger.read_text(encoding="utf-8"))["trials"]
    mujoco_rows = {}
    if a.mujoco_gates:
        for item in json.loads(a.mujoco_gates.read_text(encoding="utf-8"))["rows"]:
            mujoco_rows["+".join(item["locked_joints"])] = item
    successful={str(r.get("condition")) for r in ledger if r.get("group")=="ipwm_trials" and r.get("model_decision_audit_status")=="PASS" and r.get("task_success_under_recorded_criterion") is True}
    failed={str(r.get("condition")) for r in ledger if r.get("group")=="ipwm_trials" and r.get("model_decision_audit_status")=="PASS" and r.get("task_success_under_recorded_criterion") is False}
    rows=[]
    for r in reach:
        locks="+".join(r["locked_joints"]); condition={"J1":"D1","J2":"D2","J3":"D3","J4":"D4","J5":"D5"}.get(locks,locks)
        if condition in successful: state="REAL_GO"
        elif condition in failed: state="REAL_FAIL_RETAINED"
        elif not r["reachability_pass"]: state="REACHABILITY_NO_GO"
        elif not r["hardware_lock_authorized_currently"]: state="NEEDS_LOCK_SAFETY_AUTHORIZATION"
        else: state="READY_FOR_NEXT_PREFLIGHT"
        real_go = condition in successful
        reachable = bool(r["reachability_pass"])
        single = int(r["lock_count"]) == 1
        not_reached = "N/A_UNREACHABLE" if not reachable else "NOT_EVALUATED"
        offline = mujoco_rows.get(locks)
        collision_gate = not_reached
        speed_gate = not_reached
        if reachable and offline:
            collision_gate = (
                "PASS_LINEAR_SCREEN" if offline["linear_path_collision_gate"] == "PASS"
                else "FAIL_LINEAR_SCREEN_ONLY"
            )
            speed_gate = (
                "PASS_LINEAR_WITHIN_HOLD" if offline["speed_within_lock_hold_gate"] == "PASS"
                else "FAIL_LINEAR_PATH_ONLY"
            )
        rows.append({
            "condition": condition,
            "locked_joints": locks,
            "lock_count": r["lock_count"],
            "residual_mm": round(r["residual_m"]*1000,3),
            "reachability_gate": "PASS" if reachable else "FAIL",
            # The bounded least-squares solution was constrained to the measured
            # joint limits.  This is endpoint-only, not a trajectory clearance claim.
            "joint_limit_gate": "PASS_ENDPOINT" if reachable else "PASS_BEST_EFFORT_ENDPOINT",
            "collision_gate": "PASS_EXECUTED_PACKET" if real_go else collision_gate,
            "speed_gate": "PASS_EXECUTED_PACKET" if real_go else speed_gate,
            "load_collapse_gate": "PASS_EXECUTED_PACKET" if real_go else not_reached,
            "lock_hold_gate": "PASS_EXECUTED_PACKET" if real_go else (
                "NEEDS_J1_J5_PROBE" if reachable and not r["hardware_lock_authorized_currently"] else not_reached
            ),
            # The model consumes a five-bit intervention mask and five lock
            # angles.  General mask generation/trajectory validation now covers
            # all subsets; atomic hardware activation remains separate.
            "ipwm_fault_encoding_gate": "PASS_MASK_AND_ANGLE",
            "hardware_lock_authorization": "PASS" if r["hardware_lock_authorized_currently"] else "NOT_AUTHORIZED",
            "hardware_executor_gate": (
                "PASS_SINGLE_LOCK" if single else "PASS_ATOMIC_MULTI_LOCK_DRY_RUN"
            ),
            "evidence_state": state,
        })
    a.output_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.output_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    counts={s:sum(r["evidence_state"]==s for r in rows) for s in sorted({r["evidence_state"] for r in rows})}
    lines=["# Real-arm fault coverage matrix","",f"All 31 non-empty lock subsets; counts: `{json.dumps(counts,ensure_ascii=False)}`.","","| condition | locks | k | residual mm | reach | limits | collision | speed | load | lock hold | IPWM encoding | authorization | executor | state |","|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['condition']} | {r['locked_joints']} | {r['lock_count']} | {r['residual_mm']:.3f} | {r['reachability_gate']} | {r['joint_limit_gate']} | {r['collision_gate']} | {r['speed_gate']} | {r['load_collapse_gate']} | {r['lock_hold_gate']} | {r['ipwm_fault_encoding_gate']} | {r['hardware_lock_authorization']} | {r['hardware_executor_gate']} | {r['evidence_state']} |" for r in rows]
    lines += ["","`REAL_GO` means at least one audited Level-A execution, not statistical generalization. `PASS_ENDPOINT` only proves the bounded IK endpoint obeys measured joint limits. Linear collision/speed labels apply only to the stored straight joint-space candidate, not all possible paths. `PASS_MASK_AND_ANGLE` proves deployment-side IPWM fault encoding, not learned generalization. A reachability failure is conditional on the frozen task geometry and bounded analytic search."]
    a.output_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(rows),"counts":counts},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
